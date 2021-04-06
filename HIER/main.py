#!/usr/bin/env python
# coding: utf-8

import math, torch, torch.nn as nn, torch.nn.functional as F
import pickle as pkl, random
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from torch.autograd import Variable
import time
import gc
import os, sys
from datetime import datetime
from collections import Counter
import logging
from nltk.util import ngrams
import re, json
from tqdm import tqdm

# import spacy
# import matplotlib.pyplot as plt

from dataset import *
from utils import *
from model import *
from metrics import *
from collections import OrderedDict
from evaluate import evaluateModel
import Constants
import argparse


if not os.path.isdir('running'):
	os.makedirs('running')

def split_to_files(split):
	if split=='train':
		return train_dialog_files
	if split=='val':
		return val_dialog_files
	if split=='test':
		return test_dialog_files
	return None

def split_to_responses(split): # return original responses
	if split=='train':
		return train_responses
	if split=='val':
		return val_responses
	if split=='test':
		return test_responses
	return ValueError

def train_epoch(model, epoch, batch_size, criterion, optimizer, scheduler): # losses per batch
	model.train()
	total_loss =0
	start_time = time.time()
	ntokens = len(idxtoword)
	nbatches = len(train)//batch_size
	
#     if torch.cuda.is_available():
#         stat_cuda('before epoch')
		

	for i, (data, targets, labels) in tqdm(enumerate(data_loader(train, train_counter, batch_size, wordtoidx)), total=nbatches):

		batch_size_curr = data.shape[1]
		optimizer.zero_grad()
		output = model(data, targets)

		label_pad_mask = (labels!=0).transpose(0,1)   
			
		loss = criterion(output.view(-1, ntokens), labels.reshape(-1)) # prev code
		loss.backward()
		# # to check predicted and actual labels
		# output_max = torch.max(output, dim=2)[1]
		# for s in range(batch_size_curr):
		# 	print(output_max[:, s], '\n', labels[:, s], '\n\n')
	
#       plot_grad_flow(model.named_parameters())

		torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
		optimizer.step()
		total_loss += loss.item()*batch_size_curr
		# print(loss.item())

		elapsed = time.time()-start_time
		

	total_loss /= len(train)
	logger.debug('==> Epoch {}, Train \tLoss: {:0.4f}\tTime taken: {:0.1f}s'.format(epoch,  total_loss, elapsed))

	return total_loss




def evaluate(model, args, dataset, dataset_counter, batch_size, criterion, split, method='beam', beam_size=None):
#     print(len(dataset))
	model.eval()
	total_loss =0
	ntokens = len(wordtoidx)
	score=0
	start = time.time()

	logger.debug('{} search {}'.format(method, split))
	if method=='beam':
		logger.debug('Beam size {}'.format(beam_size))

	# .module. if using dataparallel
	with torch.no_grad():
		for i, (data, targets, labels) in tqdm(enumerate(data_loader(dataset, dataset_counter, batch_size, wordtoidx)), total=len(dataset)//batch_size):

			batch_size_curr = targets.shape[1]

			if method=='beam':
				if isinstance(model, nn.DataParallel):
					output = model.module.translate_batch(data,  beam_size , batch_size_curr)
				else:
					output = model.translate_batch(data, beam_size, batch_size_curr) #gives list of sentences itself
			elif method=='greedy':
				if isinstance(model, nn.DataParallel):
					output = model.module.greedy_search(data, batch_size_curr)
				else:
					output = model.greedy_search(data, batch_size_curr) 


			label_pad_mask = labels.transpose(0,1)!=0
			

			if torch.is_tensor(output): # greedy search
				cur_loss = criterion(output.view(-1, ntokens), labels.reshape(-1)).item()*batch_size_curr
				total_loss += cur_loss

				output = torch.max(output, dim=2)[1] # msl, batchsize
				# for s in range(batch_size_curr):
				# 	print(output[:, s],'\n', labels[:, s], '\n\n')
				# exit()

				output_max = post_process(output.transpose(0,1))				
				if i==0:
					hyp = output_max
					ref = targets.transpose(0,1)
				else:
					hyp = torch.cat((hyp, output_max), dim=0)
					ref= torch.cat((ref, targets.transpose(0,1)), dim=0)
			else: # beam search
				if i==0:
					hyp = [torch.tensor(l) for l in output]
					ref = targets.transpose(0,1)
				else:
					hyp.extend([torch.tensor(l) for l in output])
					ref= torch.cat((ref, targets.transpose(0,1)), dim=0)

		indices = list(range(0, len(dataset)))
		# indices = list(range(0, batch_size)) #uncomment this to run for one batch

		pred_hyp = tensor_to_sents(hyp, wordtoidx)
		pred_ref = split_to_responses(split)
		# pred_ref = split_to_responses(split)[:args.batch_size] # uncomment for 1 batch


		# # calculation for bleu scores of different context lengths
		# limit_small = len(dataset)//3 
		# limit_med = 2*len(dataset)//3 
		# score_small = BLEU_calc.score(hyp[:limit_small], ref[:limit_small], wordtoidx) * 100
		# score_medium = BLEU_calc.score(hyp[limit_small:limit_med], ref[limit_small:limit_med], wordtoidx)* 100
		# score_large = BLEU_calc.score(hyp[limit_med:], ref[limit_med:], wordtoidx)* 100

		# logger.debug('BLEU Scores for different buckets: ')
		# logger.debug('Small: {} \tMedium: {}\tLarge: {}'.format(score_small, score_medium, score_large))


		# using pred_hyp, pred_trg as using all dataset as indices, else use hyp,ref
		score = BLEU_calc.score(pred_hyp, pred_ref, wordtoidx) * 100
		f1_entity = F1_calc.score(pred_hyp, pred_ref, wordtoidx) * 100
		total_loss = total_loss/len(dataset)

		data, _, all_dialog_files = name_to_dataset(split)
		evaluate_dials = {}
		for i, h in enumerate(pred_hyp):
			if all_dialog_files[i] in evaluate_dials:
				evaluate_dials[all_dialog_files[i]].append(h)
			else:
				evaluate_dials[all_dialog_files[i]]=[h]
				
		# Save model predictions to json also for later evaluation
		if method=='beam':
			model_turns_file = args.log_path+'model_turns_beam_'+str(beam_size)+'_'+split+'.json'
		elif method=='greedy':
			model_turns_file = args.log_path+'model_turns_greedy_'+split+'.json'
		with open(model_turns_file, 'w') as f:
			json.dump(evaluate_dials, f)
		
		matches, successes = evaluateModel(evaluate_dials) # gives matches(inform), success

		if method=='beam':
			pred_file = open(args.log_path+'pred_beam_'+str(beam_size)+'_'+split+'.txt', 'w')
		elif method=='greedy':
			pred_file = open(args.log_path+'pred_greedy_'+split+'.txt', 'w')

		pred_file.write('\n\n***'+split+'***')
		for idx, h, r in zip(indices, pred_hyp, pred_ref):
			pred_file.write('\n\nContext: \n'+str('\n'.join(data[idx][:-1])))
			pred_file.write('\nGold sentence: '+str(r)+'\nOutput: '+str(h))
		

	elapsed = time.time()-start
	criteria = score+0.5*(matches+successes)
	logger.debug('==> {} \tLoss: {:0.4f}\tBleu: {:0.3f}\tF1-Entity {:0.3f}\tInform {:0.3f}\tSuccesses: {:0.3f}\tCriteria: {:0.3f}\tTime taken: {:0.1f}s'.format( split, total_loss, score, f1_entity, matches, successes, criteria, elapsed))
	return total_loss, score, f1_entity, matches, successes



def get_loss_nograd(model, epoch, batch_size, criterion, split): # losses per batch
	model.eval()
	total_loss =0
	start_time = time.time()
	ntokens = len(idxtoword)
	
	dataset, dataset_counter, _ = name_to_dataset(split)	
	
	with torch.no_grad():
		for i, (data, targets, labels) in enumerate(data_loader(dataset, dataset_counter, batch_size, wordtoidx)):
			batch_size_curr = data.shape[1]
			output = model(data, targets)
			label_pad_mask = (labels!=0).transpose(0,1)   			
			loss = criterion(output.view(-1, ntokens), labels.reshape(-1)) 
			total_loss += loss.item()*batch_size_curr

		elapsed = time.time()-start_time

	total_loss /= len(dataset)
	logger.debug('{} \tLoss(using ground truths): {:0.7f}\tTime taken: {:0.1f}s'.format(split, total_loss, elapsed))
	return total_loss


# stat_cuda('before training')
def training(model, args, criterion, optimizer, scheduler, optuna_callback=None):
	# global best_val_bleu, criteria, best_val_loss_ground

	best_model = None
	train_losses = []
	val_losses = []

	# if best_val_loss_ground==None: #if not set outside, initialise again
	# 	best_val_loss_ground=float("inf")
	# 	best_val_bleu=-float("inf")
	# 	criteria=-float("inf")

	best_val_loss_ground=float("inf")
	best_val_bleu=-float("inf")
	best_criteria=-float("inf")

	logger.debug('Best val loss ground at begin of training: {:0.7f}'.format(best_val_loss_ground))
	logger.debug('====> STARTING TRAINING NOW')
	val_epoch_freq=3
	
	for epoch in range(1, args.epochs + 1):
		epoch_start_time = time.time()
		train_loss = train_epoch(model, epoch, args.batch_size, criterion, optimizer, scheduler)

		val_loss_ground = get_loss_nograd(model, epoch, args.batch_size, criterion, 'val')

		train_losses.append(train_loss)
		val_losses.append(val_loss_ground)

		if val_loss_ground < best_val_loss_ground:
			best_val_loss_ground = val_loss_ground
			logger.debug('==> New optimum found wrt val loss')
			save_model(model, args, 'checkpoint_bestloss.pt',train_loss, val_loss_ground, -1)


		# for every 3 epochs, evaluate the metrics
		if epoch%val_epoch_freq!=0:
			save_model(model, args, 'checkpoint.pt', train_loss, val_loss_ground, -1)
			continue

		val_loss, val_bleu, val_f1entity, matches, successes = evaluate(model,args, val, val_counter, args.batch_size, criterion, 'val', 'greedy')
		val_criteria = val_bleu+0.5*matches+0.5*successes

		if optuna_callback is not None:
			optuna_callback(epoch/val_epoch_freq, val_criteria) # Pass the score metric on validation set here.
		
		if val_bleu > best_val_bleu:
			best_val_bleu = val_bleu
			logger.debug('==> New optimum found wrt val bleu')
			save_model(model, args, 'checkpoint_bestbleu.pt',train_loss,val_loss_ground, val_bleu)
		
		if val_criteria > best_criteria:
			best_criteria =  val_criteria
			best_model = model
			logger.debug('==> New optimum found wrt val criteria')
			save_model(model, args, 'checkpoint_criteria.pt',train_loss, val_loss_ground, val_bleu)

		save_model(model, args, 'checkpoint.pt',train_loss, val_loss_ground, val_bleu)

		scheduler.step()


	# Plot training and validation loss
	# param_range = np.arange(1, epochs+1, 1)
	# plt.plot(param_range, train_losses, label="Training loss", color="black")
	# plt.plot(param_range, val_losses, label="Cross-validation loss", color="green")
	# plt.show()

	return best_model




def save_model(model, args, name, train_loss, val_loss, val_bleu):
	checkpoint = {
					'model': model.state_dict(),
					'embedding_size': args.embedding_size,
					'nhead':args.nhead,
					'nhid': args.nhid,
					'nlayers_e1': args.nlayers_e1,
					'nlayers_e2': args.nlayers_e2,
					'nlayers_d': args.nlayers_d,
					'dropout': args.dropout				 }

	if train_loss!=-1:
		checkpoint['train_loss']=train_loss
	if val_loss!=-1:
		checkpoint['val_loss']=val_loss
	if val_bleu!=-1:
		checkpoint['val_bleu']=val_bleu

	logger.debug('==> Checkpointing everything now...in {}'.format(name))
	torch.save(checkpoint, args.log_path+name)



def load_model(model, checkpoint='checkpoint.pt'):
	global best_val_bleu, best_val_loss_ground, criteria
	load_file = checkpoint
	if os.path.isfile(load_file):
		try:
			print('Reloading previous checkpoint', load_file)

			if not torch.cuda.is_available():
				# load dataparallel model into cpu
				checkpoint = torch.load(load_file,map_location=lambda storage, loc: storage)
				new_state_dict = OrderedDict()
				for k, v in checkpoint['model'].items():
					if k[:6]=="module":
						name = k[7:] # remove `module.`
					else:
						name=k
					new_state_dict[name] = v
				# load params
				model.load_state_dict(new_state_dict)

			else:
				checkpoint = torch.load(load_file)
				model.load_state_dict(checkpoint['model'])
				# optimizer.load_state_dict(checkpoint['optim'])

			if(checkpoint.get('val_loss')):
				best_val_loss_ground = checkpoint.get('val_loss')
			else:
				best_val_loss_ground= get_loss_nograd(model, 0, args.batch_size, 'val')

			if(checkpoint.get('val_bleu')):
				best_val_bleu = checkpoint.get('val_bleu')
				logger.debug('Valid bleu of Loaded model is: {:0.4f}'.format(best_val_bleu))

			logger.debug('Loaded model, Val loss(ground): {:0.8f}'.format(best_val_loss_ground))

		except Exception as e:
			print('Loading model error')
			print(e)
	else:
		print('No model to load')



def name_to_dataset(split):
	if split=='train':
		return train, train_counter, train_dialog_files
	if split=='val':
		return val, val_counter, val_dialog_files
	if split=='test':
		return test, test_counter, test_dialog_files
	print('Error')



def testing(model, args, criterion, split, method):
	data, dataset_counter, _ = name_to_dataset(split)
	test_loss, test_bleu, test_f1entity, matches, successes = evaluate(model, args, data, dataset_counter, args.batch_size, criterion, split, method)
	return test_loss, test_bleu, test_f1entity, matches, successes

def test_split(split, model, args, criterion):
	data, dataset_counter, _ = name_to_dataset(split)
	# greedy
	evaluate(model, args, data, dataset_counter, args.batch_size, criterion, split, 'greedy')
	# beam 2
	evaluate(model, args, data, dataset_counter, args.batch_size, criterion, split, 'beam', 2)
	# beam 3
	evaluate(model, args, data, dataset_counter, args.batch_size, criterion, split, 'beam', 3)
	# beam 5
	evaluate(model, args, data, dataset_counter, args.batch_size, criterion, split, 'beam', 5)
	


def run(args, optuna_callback=None):
	global logger 

	if args.model_type=="SET":
		log_path ='running/transformer_set/'
	elif args.model_type=="HIER":
		log_path ='running/transformer_hier/'
	elif args.model_type=="MAT":
		log_path ='running/transformer_mat/'
	else:
		print('Invalid model type')
		raise ValueError

	if not os.path.isdir(log_path[:-1]):
		os.makedirs(log_path[:-1])

	args.log_path = log_path

	# file logger
	time_stamp = '{:%d-%m-%Y_%H:%M:%S}'.format(datetime.now())
	fh = logging.FileHandler(log_path+'train_'+ time_stamp  +'.log', mode='a')
	fh.setLevel(logging.DEBUG)
	fh.setFormatter(formatter)
	logger.addHandler(fh)

	# console logger - add it when running it on gpu directly to see all sentences
	ch = logging.StreamHandler()
	ch.setLevel(logging.DEBUG)
	ch.setFormatter(formatter)
	logger.addHandler(ch)
	
	logger.debug('===> \n\n' + str(args) + '\n===>\n\n')

	os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


	# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # for single device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(device)
	torch.backends.cudnn.benchmark=True
	
	max_sent_len = 50

	ntokens=len(wordtoidx)
	
	model = Transformer(ntokens, args.embedding_size, args.nhead, args.nhid, args.nlayers_e1, args.nlayers_e2, args.nlayers_d, args.dropout, args.model_type).to(device)

	criterion = nn.CrossEntropyLoss(ignore_index=0)

	seed = 123
	torch.manual_seed(seed)


	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.backends.cudnn.benchmark = True
		torch.set_default_tensor_type('torch.cuda.FloatTensor')
		# using data parallel
		model = nn.DataParallel(model, device_ids=[0,1], dim=1)
		print('putting model on cuda')
		model.to(device)
		criterion.to(device)

	print('Total number of trainable parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad)/float(1000000), 'M')

	
	optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.98)
	
	logger.debug('\n\n\n=====>\n')

	# best_val_loss_ground = load_model(model, args.log_path + 'checkpoint_criteria.pt')
	# _ = training(model, args, criterion, optimizer, scheduler, optuna_callback)
	best_val_loss_ground = load_model(model, args.log_path + 'checkpoint_criteria.pt') #load model with best criteria

	method = 'greedy'
	logger.debug('Testing model {}\n'.format(method))


	# _,test_bleu ,test_f1 ,test_matches,test_successes = testing(model, args, criterion, 'test', 'greedy')
	# logger.debug('==>Test\tBleu: {:0.3f}\tF1-Entity {:0.3f}\tInform {:0.3f}\tSuccesses: {:0.3f}'.format(test_bleu, test_f1, test_matches, test_successes))
	# logger.debug('Test critiera: {}'.format(test_bleu+0.5*(test_matches+test_successes)))

	# # To get greedy, beam(2,3,5) scores for val, test 
	# test_split('val', model, args, criterion)
	test_split('test', model, args, criterion)

	_,val_bleu ,_,val_matches,val_successes = testing(model, args, criterion, 'val', 'greedy')
	return val_bleu+0.5*(val_matches+val_successes)


# global logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")


train, train_counter, _ , train_dialog_files, train_responses  = gen_dataset_with_acts('train')
val, val_counter, _, val_dialog_files, val_responses = gen_dataset_with_acts('val')
test, test_counter, _ , test_dialog_files, test_responses = gen_dataset_with_acts('test')

# top 1500 words
idxtoword, wordtoidx = build_vocab_freqbased(load=False)
vocab_size = len(idxtoword)
ntokens = vocab_size
# print(wordtoidx)
print('length of vocab: ', vocab_size)

BLEU_calc = BLEUScorer() 
F1_calc = F1Scorer()



if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("-embed", "--embedding_size", default=100, type=int, help = "Give embedding size")
	parser.add_argument("-heads", "--nhead", default=4, type=int,  help = "Give number of heads")
	parser.add_argument("-hid", "--nhid", default=100, type=int,  help = "Give hidden size")

	parser.add_argument("-l_e1", "--nlayers_e1", default=3, type=int,  help = "Give number of layers for Encoder 1")
	parser.add_argument("-l_e2", "--nlayers_e2", default=3, type=int,  help = "Give number of layers for Encoder 2")
	parser.add_argument("-l_d", "--nlayers_d", default=3, type=int,  help = "Give number of layers for Decoder")

	parser.add_argument("-d", "--dropout",default=0.2, type=float, help = "Give dropout")
	parser.add_argument("-bs", "--batch_size", default=32 , type=int, help = "Give batch size")
	parser.add_argument("-e", "--epochs", default=30, type=int, help = "Give number of epochs")
	parser.add_argument("-model", "--model_type", default="HIER", help="Give model name one of [SET, HIER, MAT]")

	args = parser.parse_args() 
	
	run(args)

	# model = Transformer(ntokens, args.embedding_size, args.nhead, args.nhid, args.nlayers_e1, args.nlayers_e2, args.nlayers_d, args.dropout, args.model_type).to(device)
	# best_val_loss_ground = load_model(model, '../transformers/running/transformer_dyn_hdsaslide_1/checkpoint.pt')
