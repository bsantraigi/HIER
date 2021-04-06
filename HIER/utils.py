# ## multiwoz

import math, torch, torch.nn as nn, torch.nn.functional as F
import pickle as pkl, random
# from nltk.translate.bleu_score import sentence_bleu

import numpy as np
from torch.autograd import Variable
# import matplotlib.pyplot as plt
import time
import gc
import  os, sys
from datetime import datetime
from collections import Counter


max_sent_len = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

def tokenize_en(sentence):
#     return [tok.text for tok in en.tokenizer(sentence)]
    return sentence.split()

def print_tensors():
    total=0 
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                # print(type(obj),  obj.size())
                total += torch.numel(obj)*4
        except:
            pass
    
    print("{} GB".format(total/((1024**3) )))


def stat_cuda(msg):
    print('--', msg)
    print('allocated: %.2fG, max allocated: %.2fG, cached: %.2fG, max cached: %.2fG' % (
        torch.cuda.memory_allocated() / 1024 / 1024/1024,
        torch.cuda.max_memory_allocated() / 1024 / 1024/1024,
        torch.cuda.memory_cached() / 1024 / 1024/1024,
        torch.cuda.max_memory_cached() / 1024 / 1024/1024
    ))

def check_nan(t, name):
    if (t!=t).any():
        print("found nan, in ", name)


def Rand(start, end, num): 
    res = [] 
    for j in range(num): 
        res.append(random.randint(start, end))   
    return res 


# class batch from annotated transformer

def data_gen(dataset, batch_size, i, wordtoidx):
    # print(i, len(dataset))
    max_dial_len = len(dataset[i])-1
#     upper_bound = min(i+batch_size, len(dataset))
    upper_bound = i+batch_size
    vectorised_seq = []

    for d in dataset[i:upper_bound]:
#         print(len(d), end=' ')
        vectorised_seq.append([[wordtoidx.get(word, 1) for word in tokenize_en(sent)] for sent in d])

    seq_lengths = torch.LongTensor([min(len(seq), max_sent_len) for seq in vectorised_seq])
    seq_tensor = torch.zeros(batch_size, max_dial_len, max_sent_len, device=device)

    target_tensor = torch.zeros(batch_size, max_sent_len, device=device)
    label_tensor = torch.zeros(batch_size, max_sent_len, device=device)

    for idx,(seq, seqlen) in enumerate(zip(vectorised_seq, seq_lengths)):
        for i in range(seqlen-1):
            seq_tensor[idx, i, :len(seq[i])] = torch.LongTensor(seq[i])
        target_tensor[idx, :len(seq[seqlen-1])] = torch.LongTensor(seq[seqlen-1]) # last sentence in dialog
        label_tensor[idx, :len(seq[seqlen-1])-1] = torch.LongTensor(seq[seqlen-1][1:]) # last sentence in dialog from first word
    # changing labels to have SOS now, [1:]
    
    seq_tensor = seq_tensor.transpose(1,2).reshape(batch_size, -1).transpose(0,1)
    # seq_tensor - (msl*mdl , bs)

    target_tensor = target_tensor.transpose(0,1)
    label_tensor = label_tensor.transpose(0,1)

#     print(seq_tensor.size(), target_tensor.size())
    
    return seq_tensor.long(), target_tensor.long(), label_tensor.long()



def data_loader(dataset, dataset_counter, batch_size, wordtoidx): 
    # return batches according to dialog len, -> all similar at once
    # do mask also for these
    prev=0
    for dial_len, val in dataset_counter.items():
    #    if val<2:
    #        continue
        for i in range(prev, prev+val, batch_size):
#             print(i, min(batch_size, prev+val-i))
            yield data_gen(dataset,min(batch_size, prev+val-i), i, wordtoidx)
        #     break #uncomment both break statements to run for one batch
        # break

        prev += val


# class batch from annotated transformer with acts
def data_gen_acts(dataset, act_vecs, batch_size, i, wordtoidx):
    # print(i, len(dataset))
    max_dial_len = len(dataset[i])-1

    upper_bound = i+batch_size
    vectorised_seq = []
    for d in dataset[i:upper_bound]:
#         print(len(d), end=' ')
        vectorised_seq.append([[wordtoidx.get(word, 1) for word in tokenize_en(sent)] for sent in d])

    batch_actvecs = torch.tensor(act_vecs[i:upper_bound], device=device)

    seq_lengths = torch.LongTensor([min(len(seq), max_sent_len) for seq in vectorised_seq])
    seq_tensor = torch.zeros(batch_size, max_dial_len, max_sent_len, device=device)

    target_tensor = torch.zeros(batch_size, max_sent_len, device=device)
    label_tensor = torch.zeros(batch_size, max_sent_len, device=device)

    for idx,(seq, seqlen) in enumerate(zip(vectorised_seq, seq_lengths)):
        for i in range(seqlen-1):
            seq_tensor[idx, i, :len(seq[i])] = torch.LongTensor(seq[i])
        target_tensor[idx, :len(seq[seqlen-1])] = torch.LongTensor(seq[seqlen-1]) # last sentence in dialog
        label_tensor[idx, :len(seq[seqlen-1])-1] = torch.LongTensor(seq[seqlen-1][1:]) # last sentence in dialog from first word, ie without sos
    
    seq_tensor = seq_tensor.transpose(1,2).reshape(batch_size, -1).transpose(0,1)
    # seq_tensor - (msl*mdl , bs)

    target_tensor = target_tensor.transpose(0,1)
    label_tensor = label_tensor.transpose(0,1)
    batch_actvecs = batch_actvecs.transpose(0,1)

    # print(batch_actvecs.shape)
    
    return seq_tensor.long(), target_tensor.long(), label_tensor.long(), batch_actvecs.float()


def data_loader_acts(dataset, dataset_counter, act_vecs, batch_size, wordtoidx): 
    # return batches according to dialog len, -> all similar at once
    # do mask also for these
    prev=0
    for dial_len, val in dataset_counter.items():
    #    if val<2:
    #        continue
        for i in range(prev, prev+val, batch_size):
#             print(i, min(batch_size, prev+val-i))
            yield data_gen_acts(dataset, act_vecs, min(batch_size, prev+val-i), i, wordtoidx)
        #     break # uncomment both break stats to run for 1 batch for SET++,HIER++ models
        # break
        prev += val



def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.norm())
#             print(n, p.grad)
            if p.grad.abs().max()==0.0:
                print('grad became zero: ',n)

    # plt.figure(figsize=(16, 20))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical", fontsize=6)
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
#     plt.tight_layout()
#     plt.savefig("temp.png")
    plt.show()
