## SET, HIER, MAT

import math, torch, torch.nn as nn, torch.nn.functional as F
from Beam import Beam
import numpy as np
import Constants

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

def _gen_mask(src_len, tgt_len):        
	t = torch.zeros(src_len, src_len, device=device)
	f = torch.ones(tgt_len, tgt_len, device=device)
	for i in range(0, src_len, tgt_len):
		t[i:i+tgt_len, i:i+tgt_len]=f
	t = t.float().masked_fill(t==0, float('-inf')).masked_fill(t==1, 0)
	return t

def _gen_mask_sent(sz):
	mask = ((torch.triu(torch.ones(sz, sz, device=device)) == 1) * 1.0).transpose(0,1)
	mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1, 0)    
	return mask


def _generate_square_subsequent_mask(sz):
	mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1) 
	# triu returns upper triangular part of matrix, zeroes others
	mask = mask.float().masked_fill(mask == 1, float('-inf')) # float('-inf')
	return mask # upper tri. zero and lower half is -inf

def _gen_mask_bidirec(src_len, tgt_len):        
	t = torch.zeros(src_len, src_len, device=device)
	for i in range(0, src_len, tgt_len):
		t[i:i+tgt_len, :i+tgt_len]= torch.ones(tgt_len, i+tgt_len, device=device) 
	t = t.float().masked_fill(t==0, float('-inf')).masked_fill(t==1, 0)
	return t

def _gen_mask_hierarchical(src_len, tgt_len):        
    t = torch.zeros(src_len, src_len, device=device)
    f = torch.ones(tgt_len, tgt_len, device=device)
    for i in range(0, src_len, tgt_len):
        t[i:i+tgt_len, i:i+tgt_len]=f
        t[i:i+tgt_len, -tgt_len:] = f
    t = t.float().masked_fill(t==0, float('-inf')).masked_fill(t==1, 0)
    return t

def post_process(output_max): # keeps till eos, after that all changed to pad
	# output_max.shape - (bs, msl)
	eos_index = 3 # wordtoidx['EOS']
	mask_len = [(line==eos_index).nonzero()[0]+1 if (line==eos_index).any() else output_max.shape[1] for line in output_max ]

	mask_values = []
	try:
		for idx,e in enumerate(mask_len):
			mask_values.extend(range(idx*output_max.shape[1], e + idx*output_max.shape[1]))
	except:
		print('error here ', e)
		print(mask_len)
		
	mask =  torch.zeros((output_max.reshape(-1).shape[0]), device=device)
	mask[mask_values]=1
	mask = mask.view(*output_max.shape)
	output_max = output_max * mask
#     output = output  # to do? - (bs, msl, embed) - replace with pad embeddings after first eos
	return output_max

def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
	''' Collect tensor parts associated to active instances. '''

	_, *d_hs = beamed_tensor.size()
	n_curr_active_inst = len(curr_active_inst_idx)
	new_shape = (n_curr_active_inst * n_bm, *d_hs)

	beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
	beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
	beamed_tensor = beamed_tensor.view(*new_shape)

	return beamed_tensor

def get_inst_idx_to_tensor_position_map(inst_idx_list):
	''' Indicate the position of an instance in a tensor. '''
	return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}



class PositionalEncoding(nn.Module):
	def __init__(self, d_model, dropout, max_len= 5000):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)#size = (max_len, 1)
		
		if d_model%2==0:
			div_term = torch.exp(torch.arange(0, d_model,2).float() * (-math.log(10000.0)/d_model))
			pe[:, 0::2] = torch.sin(position * div_term)
			pe[:, 1::2] = torch.cos(position*div_term)
		else:
			div_term = torch.exp(torch.arange(0, d_model+1, 2).float() * (-math.log(10000.0)/d_model))
			pe[:, 0::2] = torch.sin(position * div_term)
			pe[:, 1::2] = torch.cos(position * div_term[:-1])

		pe = pe.unsqueeze(1) # size - (max_len, 1, d_model)
		self.register_buffer('pe', pe)
		# print('POS ENC. :', pe.size()) # 5000,1,embed_size
	
	def forward(self, x): # 1760xbsxembed
		x = x+self.pe[:x.size(0), :, :].repeat(1, x.size(1), 1)
		return self.dropout(x)


class Transformer(nn.Module):
	def __init__(self, ntoken, ninp, nhead, nhid, nlayers_e1, nlayers_e2, nlayers_d, dropout, ablation='SET'):
		# ninp is embed_size
		super(Transformer, self).__init__()
		from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
		
		encoder_layers1 = TransformerEncoderLayer(ninp, nhead, nhid, dropout) ## sizes
		self.transformer_encoder=TransformerEncoder(encoder_layers1, nlayers_e1)

		encoder_layers2 = TransformerEncoderLayer(ninp, nhead, nhid, dropout, activation='relu')
		self.transformer_encoder_sent = TransformerEncoder(encoder_layers2, nlayers_e2)
		
		decoder_layers = TransformerDecoderLayer(ninp, nhead, nhid, dropout, activation='relu')
		self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers_d)
		
		self.encoder = nn.Embedding(ntoken, ninp)
		self.decoder = nn.Linear(ninp, ntoken)
		self.pos_encoder = PositionalEncoding(ninp, dropout)
		
		self.ninp = ninp       
		
		# self.max_sent_len = tgt_mask.size(0)
		self._reset_parameters()

		if ablation=='SET':
			self.mask_func = _gen_mask
		elif (ablation=='HIER' or ablation=='MAT'):
			self.mask_func = _gen_mask_hierarchical
		else:
			print('Not a valid ablation')
			raise ValueError

		self.ablation = ablation

	def _reset_parameters(self):
		r"""Initiate parameters in the transformer model."""
		for n, p in self.named_parameters():
			if p.dim() > 1:
				# print(n)
				torch.nn.init.xavier_normal_(p)
	
	def init_weights(self):
		initrange = 0.1
		self.encoder.weight.data.uniform_(-initrange, initrange)
		self.decoder.bias.data.zero_()
		self.decoder.weight.data.uniform_(-initrange, initrange)

	
	def forward(self, src, tgt, src_pad_mask=None, tgt_pad_mask=None):
		max_sent_len = 50
		src_mask = torch.zeros(max_sent_len,max_sent_len, device=device)
		tgt_mask = _gen_mask_sent(tgt.shape[0])		
		batch_size = tgt.shape[1]
		
		max_dial_len = src.reshape(max_sent_len, -1, batch_size).shape[1]
		
		src_sent = src.reshape(max_sent_len, -1, batch_size).transpose(0,1).reshape(-1, batch_size)
		src_pad_mask_sent= (src_sent==0).transpose(0,1)
		
		# this mask depends on mdl, make dynamically
		# for SET
		# src_mask_sent = _gen_mask(max_dial_len*max_sent_len, max_sent_len) 
		# for HIER
		# src_mask_sent = _gen_mask_hierarchical(max_dial_len*max_sent_len, max_sent_len) # this one is bidirectional 
		src_mask_sent = self.mask_func(max_dial_len*max_sent_len, max_sent_len)

		src = src.reshape(max_sent_len, -1)

		src_pad_mask = (src==0).transpose(0,1)
		tgt_pad_mask = (tgt==0).transpose(0,1)

		src = self.encoder(src) * math.sqrt(self.ninp)
		src = self.pos_encoder(src)

		# encoder 1
		if self.ablation=='SET' or self.ablation=='HIER':
			memory_inter = self.transformer_encoder(src, src_mask, src_pad_mask)
		elif self.ablation=='MAT':
			memory_inter = src
#         check_nan(memory_inter, 'memory_inter')
		memory_inter = memory_inter.view(max_sent_len, -1, batch_size, self.ninp).transpose(0,1).reshape(-1, batch_size, self.ninp)

		# encoder 2
		memory_inter = self.pos_encoder(memory_inter)
		memory = self.transformer_encoder_sent(memory_inter, src_mask_sent, src_pad_mask_sent)

		# check_nan(memory, 'memory')
		
		# decoder
		tgt = self.encoder(tgt) * math.sqrt(self.ninp)
		tgt = self.pos_encoder(tgt)
		output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask)
		
		# check_nan(output, 'output')

		output = self.decoder(output)
		return output

	def greedy_search(self, src, batch_size):
		max_sent_len = 50
		max_dial_len = src.reshape(max_sent_len, -1, batch_size).shape[1]
		tgt = 2*torch.ones(1, batch_size , device=device).long()

		for i in range(1, max_sent_len+1):
			output = self.forward(src, tgt)[-1,:,:].unsqueeze(0)
			# print('output ', output.shape) # i,bs,vocab
			if i==1:
				logits = output
			else:
				logits = torch.cat([logits, output], dim=0)
			output_max = torch.max(output, dim=2)[1]
			tgt = torch.cat([tgt, output_max], dim=0)

		tgt = tgt[1: , :]
		return logits

		
	def translate_batch(self, src, n_bm, batch_size): # , src_pad_mask, tgt_pad_mask
		# adopted from HDSA_Dialog
		device = src.device

		max_sent_len = 50
		max_dial_len = src.reshape(max_sent_len, -1, batch_size).shape[1]

		src = src.transpose(0,1) # src shape changed to (bs*mdl, msl)


		def collate_active_info(src, inst_idx_to_position_map, active_inst_idx_map):
			# Sentences which are still active are collected,
			# so the decoder will not run on completed sentences.
			n_prev_active_inst = len(inst_idx_to_position_map)
			active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
			active_inst_idx = torch.LongTensor(active_inst_idx).to(device)
		
			active_src_seq = collect_active_part(src, active_inst_idx, n_prev_active_inst, n_bm)                
			active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)       
			return active_src_seq, active_inst_idx_to_position_map
			
		def beam_decode_step(inst_dec_beams, len_dec_seq, active_inst_idx_list, src,                              inst_idx_to_position_map, n_bm):
			''' Decode and update beam status, and then return active beam idx '''
			n_active_inst = len(inst_idx_to_position_map)
				
			dec_partial_seq = [inst_dec_beams[idx].get_current_state() 
							   for idx in active_inst_idx_list if not inst_dec_beams[idx].done]
			dec_partial_seq = torch.stack(dec_partial_seq).to(device)
#             print(dec_partial_seq.shape) #32,5,1
			dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
#             print(dec_partial_seq.shape) #1, 150
			
			# print( src.shape, dec_partial_seq.shape) # src is 50, 150
			logits = self.forward(src.transpose(0,1) , dec_partial_seq.transpose(0,1))[-1, :, :].unsqueeze(0) # error here

			# print(logits.shape)
			word_prob = F.log_softmax(logits, dim=2)
			word_prob = word_prob.view(n_active_inst, n_bm, -1) # active, bms, vocab 
			# print('word_prob shape ', word_prob.shape) # should remain same for all steps!!
			
			# print(inst_idx_to_position_map) # 0:0, 1:1 map

			# Update the beam with predicted word prob information and collect incomplete instances
			active_inst_idx_list = []
			for inst_idx, inst_position in inst_idx_to_position_map.items():
				is_inst_complete = inst_dec_beams[inst_idx].advance(word_prob[inst_position]) # gotta check advance method here!!
				if not is_inst_complete:
					active_inst_idx_list += [inst_idx]
		
			return active_inst_idx_list
			
		with torch.no_grad():
			# repeat src n_bm times

			src = src.repeat(1, n_bm).reshape(batch_size*n_bm , -1)
			#  bm*batch_size*mdl, msl
			
			inst_dec_beams = [Beam(n_bm, device=device) for _ in range(batch_size)]
			active_inst_idx_list = list(range(batch_size))
			inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)
			
			for len_dec_seq in range(1, 50+1):
				active_inst_idx_list = beam_decode_step(inst_dec_beams, len_dec_seq, active_inst_idx_list, src, inst_idx_to_position_map, n_bm)
				if not active_inst_idx_list:
					break
				src, inst_idx_to_position_map = collate_active_info(src, inst_idx_to_position_map, active_inst_idx_list)
				
			def collect_hypothesis_and_scores(inst_dec_beams, n_best):
				all_hyp, all_scores = [], []
				for beam in inst_dec_beams:
					scores = beam.scores
					hyps = np.array([beam.get_hypothesis(i) for i in range(beam.size)], 'long')
					lengths = (hyps != Constants.PAD).sum(-1)
					normed_scores = [scores[i].item()/lengths[i] for i, hyp in enumerate(hyps)]
					idxs = np.argsort(normed_scores)[::-1]

					all_hyp.append([hyps[idx] for idx in idxs])
					all_scores.append([normed_scores[idx] for idx in idxs])
				return all_hyp, all_scores

			batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, n_bm)
			
		batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, n_bm)
		
		result = []
		for _ in batch_hyp:
			finished = False
			for r in _:
				if len(r) >= 8 and len(r) < max_sent_len:
					result.append(r)
					finished = True
					break
			if not finished:
				result.append(_[0])
		return result



class Transformer_acts(nn.Module):
	def __init__(self, ntoken, ninp, nhead, nhid, nlayers_e1, nlayers_e2, nlayers_d, dropout, ablation):
		# ninp is embed_size
		super(Transformer_acts, self).__init__()
		from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
		
		encoder_layers1 = TransformerEncoderLayer(ninp, nhead, nhid, dropout) ## sizes
		self.transformer_encoder=TransformerEncoder(encoder_layers1, nlayers_e1)

		encoder_layers2 = TransformerEncoderLayer(ninp, nhead, nhid, dropout, activation='relu')
		self.transformer_encoder_sent = TransformerEncoder(encoder_layers2, nlayers_e2)
		
		decoder_layers = TransformerDecoderLayer(ninp, nhead, nhid, dropout, activation='relu')
		self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers_d)
		
		self.encoder = nn.Embedding(ntoken, ninp)
		self.decoder = nn.Linear(ninp, ntoken)
		self.pos_encoder = PositionalEncoding(ninp, dropout)
		
		self.ninp = ninp

		self.act_embedding = nn.Linear(44, self.ninp) # Comment this for transformers_dyn_hdsaslide_1 checkpoint

		if ablation=='SET++':
			self.mask_func = _gen_mask
		elif ablation=='HIER++':
			self.mask_func = _gen_mask_hierarchical
		else:
			print('Not a valid ablation')
			raise ValueError

		self.ablation = ablation
		
		# self.max_sent_len = tgt_mask.size(0)
		self._reset_parameters()

	def _reset_parameters(self):
		r"""Initiate parameters in the transformer model."""
		for n, p in self.named_parameters():
			if p.dim() > 1:
				# print(n)
				torch.nn.init.xavier_normal_(p)
	
	def init_weights(self):
		initrange = 0.1
		self.encoder.weight.data.uniform_(-initrange, initrange)
		self.decoder.bias.data.zero_()
		self.decoder.weight.data.uniform_(-initrange, initrange)
	
	def forward(self, src, tgt, act_vecs, src_pad_mask=None, tgt_pad_mask=None):
		max_sent_len = 50
		src_mask = torch.zeros(max_sent_len,max_sent_len, device=device)
		tgt_mask = _gen_mask_sent(tgt.shape[0])		
		batch_size = tgt.shape[1]
		
		max_dial_len = src.reshape(max_sent_len, -1, batch_size).shape[1]
		
		src_sent = src.reshape(max_sent_len, -1, batch_size).transpose(0,1).reshape(-1, batch_size)
		src_pad_mask_sent= (src_sent==0).transpose(0,1)
		
		# this mask depends on mdl, make dynamically
		# src_mask_sent = _gen_mask_hierarchical(max_dial_len*max_sent_len, max_sent_len) # this mask focuses on utterances like te1
		# src_mask_sent = _gen_mask(max_dial_len*max_sent_len) # this mask is unidirectional
		src_mask_sent = self.mask_func(max_dial_len*max_sent_len, max_sent_len)

		src = src.reshape(max_sent_len, -1)

		src_pad_mask = (src==0).transpose(0,1)
		tgt_pad_mask = (tgt==0).transpose(0,1)

		src = self.encoder(src) * math.sqrt(self.ninp)
		src = self.pos_encoder(src)

		# encoder 1
		memory_inter = self.transformer_encoder(src, src_mask, src_pad_mask)

#         check_nan(memory_inter, 'memory_inter')

		memory_inter = memory_inter.view(max_sent_len, -1, batch_size, self.ninp).transpose(0,1).reshape(-1, batch_size, self.ninp)

		# encoder 2
		memory_inter = self.pos_encoder(memory_inter)
		memory = self.transformer_encoder_sent(memory_inter, src_mask_sent, src_pad_mask_sent)

		# check_nan(memory, 'memory')
		
		# decoder - tgt shape - (msl, batch_size, embed)- add act_vec of (None ,bs,embed)
		tgt = self.encoder(tgt) * math.sqrt(self.ninp)
		# act_vecs.T is batch_size, 44 ->embed of 100

		tgt = self.pos_encoder(tgt) + self.act_embedding(act_vecs.transpose(0,1)).unsqueeze(0)
		output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask)
		
		# check_nan(output, 'output')

		output = self.decoder(output)
		return output

	def greedy_search(self, src, act_vecs, batch_size):
		max_sent_len = 50
		max_dial_len = src.reshape(max_sent_len, -1, batch_size).shape[1]
		tgt = 2*torch.ones(1, batch_size , device=device).long()
		eos_tokens = 3*torch.ones(1, batch_size, device=device).long()

		for i in range(1, max_sent_len+1): # predict 48 words + sos+eos=50
			output = self.forward(src, tgt, act_vecs)[-1,:,:].unsqueeze(0)
			# print('output ', output.shape) # i, bs, vocab
			if i==1:
				logits = output
			else:
				logits = torch.cat([logits, output], dim=0)
			output_max = torch.max(output, dim=2)[1]
			tgt = torch.cat([tgt, output_max], dim=0)

		tgt = torch.cat([tgt[:49,:], eos_tokens], dim=0)
		return logits, tgt

		
	def translate_batch(self, src, act_vecs, n_bm, batch_size): # , src_pad_mask, tgt_pad_mask
		# adopted from HDSA_Dialog
		device = src.device

		max_sent_len = 50
		max_dial_len = src.reshape(max_sent_len, -1, batch_size).shape[1]

		src = src.transpose(0,1) # src shape changed to (bs*mdl, msl)
		act_vecs = act_vecs.transpose(0, 1) # act_vecs changed to bs,44


		def collate_active_info(src, act_vecs, inst_idx_to_position_map, active_inst_idx_map):
			# Sentences which are still active are collected,
			# so the decoder will not run on completed sentences.
			n_prev_active_inst = len(inst_idx_to_position_map)
			active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
			active_inst_idx = torch.LongTensor(active_inst_idx).to(device)
		
			active_src_seq = collect_active_part(src, active_inst_idx, n_prev_active_inst, n_bm)      
			active_act_vecs = collect_active_part(act_vecs, active_inst_idx, n_prev_active_inst, n_bm)

			active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)       
			return active_src_seq, active_act_vecs, active_inst_idx_to_position_map
			
		def beam_decode_step(inst_dec_beams, len_dec_seq, active_inst_idx_list, src,act_vecs, inst_idx_to_position_map, n_bm):
			''' Decode and update beam status, and then return active beam idx '''
			n_active_inst = len(inst_idx_to_position_map)
				
			dec_partial_seq = [inst_dec_beams[idx].get_current_state() 
							   for idx in active_inst_idx_list if not inst_dec_beams[idx].done]
			dec_partial_seq = torch.stack(dec_partial_seq).to(device)
			dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
			
			# print( src.shape, dec_partial_seq.shape , act_vecs.shape) # src is 50, 150
			logits = self.forward(src.transpose(0,1) , dec_partial_seq.transpose(0,1), act_vecs.transpose(0, 1))[-1, :, :].unsqueeze(0) # error here

			# print(logits.shape)
			word_prob = F.log_softmax(logits, dim=2)
			word_prob = word_prob.view(n_active_inst, n_bm, -1) # active, bms, vocab 
			
			# print(inst_idx_to_position_map) # 0:0, 1:1 map

			# Update the beam with predicted word prob information and collect incomplete instances
			active_inst_idx_list = []
			for inst_idx, inst_position in inst_idx_to_position_map.items():
				is_inst_complete = inst_dec_beams[inst_idx].advance(word_prob[inst_position]) # gotta check advance method here!!
				if not is_inst_complete:
					active_inst_idx_list += [inst_idx]
		
			return active_inst_idx_list
			
		with torch.no_grad():
			# repeat src n_bm times
			# act_vecs shape is bs,44 after T

			src = src.repeat(1, n_bm).reshape(batch_size*n_bm , -1) # bm*batch_size, msl*mdl
			act_vecs = act_vecs.repeat(1, n_bm).reshape(batch_size*n_bm, -1)
			# act_vecs -> bs*n_bm, 44

			
			inst_dec_beams = [Beam(n_bm, device=device) for _ in range(batch_size)]
			active_inst_idx_list = list(range(batch_size))
			inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)
			
			for len_dec_seq in range(1, max_sent_len+1):
				active_inst_idx_list = beam_decode_step(inst_dec_beams, len_dec_seq, active_inst_idx_list, src, act_vecs,  inst_idx_to_position_map, n_bm)
				if not active_inst_idx_list:
					break
				src, act_vecs,  inst_idx_to_position_map = collate_active_info(src, act_vecs, inst_idx_to_position_map, active_inst_idx_list)
				
			def collect_hypothesis_and_scores(inst_dec_beams, n_best):
				all_hyp, all_scores = [], []
				for beam in inst_dec_beams:
					scores = beam.scores
					hyps = np.array([beam.get_hypothesis(i) for i in range(beam.size)], 'long')
					lengths = (hyps != Constants.PAD).sum(-1)
					normed_scores = [scores[i].item()/lengths[i] for i, hyp in enumerate(hyps)]
					idxs = np.argsort(normed_scores)[::-1]

					all_hyp.append([hyps[idx] for idx in idxs])
					all_scores.append([normed_scores[idx] for idx in idxs])
				return all_hyp, all_scores

			batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, n_bm)
			
		batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, n_bm)
		
		result = []
		for _ in batch_hyp:
			finished = False
			for r in _:
				if len(r) >= 8 and len(r) < max_sent_len:
					result.append(r)
					finished = True
					break
			if not finished:
				result.append(_[0])
		return result
	

	