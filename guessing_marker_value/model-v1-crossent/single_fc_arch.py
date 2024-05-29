import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import pandas

import lightning.pytorch as pl

DEVICE = "cuda"

class EmbedLayer(nn.Module):
	def __init__(self, dic_size, embedding_dim, num_embeddings):
		super().__init__()
		self.embedding_dim = embedding_dim
		self.bin_embedding = nn.Embedding(num_embeddings, embedding_dim=embedding_dim)
		self.marker_embedding = nn.Embedding(dic_size, embedding_dim=embedding_dim)

		# with torch.no_grad():
		# 	self.bin_embedding.weight[-1] = torch.zeros(embedding_dim)

	def forward(self, bins, markers):
		name_embeds = self.marker_embedding(markers)
		value_embeds = self.bin_embedding(bins)

		return name_embeds + value_embeds

# %%
class FeedForward(torch.nn.Module):
	def __init__(self, d_embed, ff_dim):
		super().__init__()
		self.d_embed = d_embed
		self.w_1 = torch.nn.Linear(d_embed, ff_dim)
		self.w_2 = torch.nn.Linear(ff_dim, d_embed)
		self.m = torch.nn.ReLU()

	def forward(self, x):
		assert len(x.shape) == 3 # batch, seq, d_model
		assert x.shape[-1] == self.d_embed

		x = self.w_1(x)
		x = self.m(x)
		x = self.w_2(x)

		assert len(x.shape) == 3  # batch, seq, d_model
		assert x.shape[-1] == self.d_embed
		return x

# %%
class EncoderLayer(torch.nn.Module):
	def __init__(self, d_embed, num_heads, d_ff, attn_dropout):
		super().__init__()

		self.to_q = torch.nn.Linear(d_embed, d_embed)
		self.to_k = torch.nn.Linear(d_embed, d_embed)
		self.to_v = torch.nn.Linear(d_embed, d_embed)

		self.attention = torch.nn.MultiheadAttention(embed_dim=d_embed, num_heads=num_heads, batch_first=True, dropout=attn_dropout)
		self.ff = FeedForward(d_embed, d_ff)
		self.norm1 = nn.LayerNorm(d_embed)
		self.norm2 = nn.LayerNorm(d_embed)

	def forward(self, x, attn_mask=None):
		q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

		x_tmp1 = x
		x, att_weights = self.attention(q, k, v, attn_mask = attn_mask, average_attn_weights=False)
		x = x + x_tmp1
		x = self.norm1(x)

		x_tmp2 = x
		x = self.ff(x)
		x = x + x_tmp2
		x = self.norm2(x)

		return x, att_weights

# %%
class Classifier(torch.nn.Module):
	def __init__(self, d_embed, num_bins):
		super().__init__()
		self.relu = nn.ReLU()
		self.fc1 = nn.Linear(d_embed, 256)
		self.fc2 = nn.Linear(256, 256)
		self.fc3 = nn.Linear(256, num_bins-1)

	def forward(self, x):
		x = self.fc1(x)
		x = self.fc2(self.relu(x))
		x = self.fc3(self.relu(x))
		return x

# %%
class Model(pl.LightningModule):
	def __init__(self, 
			  dic_size, 
			  seq_size, 
			  num_bins, 
			  d_embed, 
			  d_ff, 
			  num_heads, 
			  num_layers, 
			  attn_dropout=0.0, 
			  lr=0.00001, 
			  attn_mask=None, 
			  masking="single"):
		super().__init__()
		self.embedding = EmbedLayer(dic_size=dic_size, embedding_dim=d_embed, num_embeddings=num_bins)
		self.enc_layers = torch.nn.ModuleList([
				EncoderLayer(d_embed, num_heads, d_ff, attn_dropout) for i in range(num_layers)
		])
		self.fc = Classifier(d_embed, num_bins)

		self.dic_size = dic_size
		self.seq_size = seq_size
		self.num_bins = num_bins

		self.d_embed = d_embed
		self.d_ff = d_ff
		self.num_heads = num_heads
		self.num_layers = num_layers

		self.attn_dropout = attn_dropout
		self.lr = lr

		self.attn_mask = nn.Parameter(attn_mask, requires_grad=False) if attn_mask else None
		self.masking = masking

		'''Used for logging and tracking'''
		self.correct_outputs = []
		self.dset_size = 0
		self.epoch = 0

		self.save_hyperparameters()

	def configure_optimizers(self):
		optimizer = torch.optim.AdamW(list(self.embedding.parameters()) + 
									list(self.enc_layers.parameters()) +
									list(self.fc.parameters()),
									lr=0.00001)
		return optimizer

	def perform_loss_step(self, batch, mode='train'):
		bins, markers = batch

		#### single ####
		if self.masking == "single":
			missing_ids = torch.randint(0, self.seq_size, (bins.shape[0],))
			labels = bins[torch.arange(bins.shape[0]), missing_ids] 
			bins[torch.arange(bins.shape[0]), missing_ids] = self.num_bins-1

			attn_mask = None
			if self.attn_mask is None:
				attn_mask_base = F.one_hot(missing_ids, num_classes=self.dic_size).bool() 
				attn_mask = torch.cat([attn_mask_base[i].repeat(self.num_heads, self.seq_size, 1) for i in range(attn_mask_base.shape[0])]).to(self.device) # batch_size x seq_size x seq_size
			else:
				attn_mask = self.attn_mask

			embedded = self.embedding(bins, markers)
			encoded = embedded
			for enc_layer in self.enc_layers:
				encoded, _ = enc_layer(encoded, attn_mask)

			all_logits = self.fc(encoded)
			pred_logits = all_logits[torch.arange(bins.shape[0]), missing_ids, :]

			preds = torch.argmax(pred_logits, dim=-1)	
			correct = torch.sum(preds == labels)
			loss = torch.nn.CrossEntropyLoss()(pred_logits, labels)

			self.log(mode+'_loss', loss)

			self.correct_outputs.append(correct)
			self.dset_size += bins.shape[0]

			return loss

		else:
			#### multi ####
			mask_how_many = (self.epoch // 100) + 1
			missing_ids = torch.stack([torch.randperm(self.seq_size) for _ in range(bins.shape[0])])[:, :mask_how_many].flatten()
			element_ids = torch.arange(bins.shape[0]).repeat_interleave(mask_how_many)
			labels = bins[element_ids, missing_ids]
			bins[element_ids, missing_ids] = self.num_bins-1

			attn_mask = None
			if self.attn_mask is None:
				attn_mask_base = torch.zeros(bins.shape).bool() #batch_size x seq_size x seq_size
				attn_mask_base[element_ids, missing_ids] = True
				attn_mask = torch.cat([attn_mask_base[i].repeat(self.num_heads, self.seq_size, 1) for i in range(attn_mask_base.shape[0])]).to(self.device) # batch_size x seq_size x seq_size
			else:
				attn_mask = self.attn_mask

			embedded = self.embedding(bins, markers)
			encoded = embedded
			for enc_layer in self.enc_layers:
				encoded, _ = enc_layer(encoded, attn_mask)

			all_logits = self.fc(encoded)
			pred_logits = all_logits[element_ids, missing_ids, :]

			preds = torch.argmax(pred_logits, dim=-1)	
			correct = torch.sum(preds == labels) // mask_how_many
			loss = torch.nn.CrossEntropyLoss()(pred_logits, labels)

			self.log(mode+'_loss', loss)

			self.correct_outputs.append(correct)
			self.dset_size += bins.shape[0]

			return loss

	def training_step(self, batch, batch_idx):
		return self.perform_loss_step(batch, mode='train')

	def validation_step(self, batch, batch_idx):
		return self.perform_loss_step(batch, mode='val')
	
	def on_train_epoch_end(self):
		correct_classified = torch.stack(self.correct_outputs).sum()
		epoch_mean = correct_classified / self.dset_size

		self.log("training_epoch_acc", epoch_mean)

		self.correct_outputs.clear()
		self.dset_size = 0
		self.epoch += 1

	def predict(self, x):
		'''
		Missing markers for every example have to be passed additionaly
		for prediction.
		'''
		bins, markers, missing_ids = x

		att_weights_all_layers = []

		encoded = self.embedding(bins, markers)
		for enc_layer in self.enc_layers:
			encoded, att_weights = enc_layer(encoded, self.attn_mask)
			att_weights_all_layers.append(att_weights)
				
		all_logits = self.fc(encoded)
		predicted_logits = all_logits[torch.arange(bins.shape[0]), missing_ids, :]
				
		return predicted_logits, att_weights_all_layers