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

def get_positional_encoding(seq_len, embedding_dim):
	v = torch.arange(0, seq_len) * 1.0
	base = (torch.ones(embedding_dim)/10000)
	poww = torch.arange(0, embedding_dim) // 2 * 2.0 / embedding_dim
	denom = torch.pow(base, poww)
	res = torch.matmul(torch.reshape(v, (seq_len,1)), torch.reshape(denom, (1,embedding_dim)))

	res[::2, :] = torch.sin(res[::2, :])
	res[1::2, :] = torch.cos(res[1::2, :])
	return res.clone().float()

class EmbedLayer(nn.Module):
	def __init__(self, dic_size=40, embedding_dim=64, num_embeddings=6):
			super().__init__()
			self.embedding_dim = embedding_dim
			# self.name_embedding = nn.Parameter(torch.zeros(size=(dic_size, embedding_dim)))
			self.value_embedding = nn.Embedding(num_embeddings, embedding_dim=embedding_dim)
			# self.mask_embedding = nn.Parameter(torch.rand(size=(embedding_dim,)))
			self.positional_embedding = nn.Parameter(get_positional_encoding(dic_size, embedding_dim), requires_grad=True)
			self.pred_embedding = nn.Parameter(torch.zeros(self.embedding_dim), requires_grad=False)

	def forward(self, x, y):
			name_embeds = self.positional_embedding.repeat(x.shape[0], 1, 1)
			# mask_embeds = self.mask_embedding.repeat(x.shape[0], 1)
			# x[torch.arange(x.shape[0]), y] = 6
			value_embeds = self.value_embedding(x)
			value_embeds[torch.arange(x.shape[0]), y, :] = self.pred_embedding.repeat(x.shape[0], 1)
			# value_embeds[torch.arange(x.shape[0]), y, :] = mask_embeds

			return name_embeds + value_embeds
			# return value_embeds

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
	def __init__(self, d_embed, num_heads, d_ff, attn_mask=None):
		super().__init__()

		self.to_q = torch.nn.Linear(d_embed, d_embed)
		self.to_k = torch.nn.Linear(d_embed, d_embed)
		self.to_v = torch.nn.Linear(d_embed, d_embed)

		self.attention = torch.nn.MultiheadAttention(embed_dim=d_embed, num_heads=num_heads, batch_first=True)
		self.ff = FeedForward(d_embed, d_ff)
		self.norm1 = nn.LayerNorm(d_embed)
		self.norm2 = nn.LayerNorm(d_embed)

		self.attn_mask = nn.Parameter(attn_mask, requires_grad=False)

	def forward(self, x):
		q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

		x_tmp1 = x

		# print(self.attn_mask)

		x, _ = self.attention(q, k, v, attn_mask = self.attn_mask)
		x = x + x_tmp1
		x = self.norm1(x)

		x_tmp2 = x
		x = self.ff(x)
		x = x + x_tmp2
		x = self.norm2(x)

		return x

# %%
class Classifier(torch.nn.Module):
	def __init__(self, d_embed, num_bins):
		super().__init__()

		# self.linear = nn.Linear(d_embed, num_bins)
		# self.act = nn.GELU()
		# self.norm = nn.LayerNorm(num_bins)
		# self.softmax = torch.nn.LogSoftmax(dim=-1)

		self.dropout = nn.Dropout(0.1)
		self.relu =  nn.ReLU()
		self.fc1 = nn.Linear(d_embed, 256)
		self.fc2 = nn.Linear(256, num_bins)
		self.softmax = nn.LogSoftmax(dim=-1)


	def forward(self, x):
		# return self.softmax(self.norm(self.act(self.linear(x))))
		return self.softmax(self.fc2(self.relu(self.fc1(x))))

# %%
class Model(pl.LightningModule):
	def __init__(self, dic_size, num_bins, d_embed, d_ff, num_heads, num_layers, attn_mask=None):
		super().__init__()
		self.embedding = EmbedLayer(dic_size=dic_size, embedding_dim=d_embed, num_embeddings=num_bins)
		self.enc_layers = torch.nn.ModuleList([
				EncoderLayer(d_embed, num_heads, d_ff, attn_mask=attn_mask) for i in range(num_layers)
		])
		self.fc = Classifier(d_embed, num_bins)

		self.correct_outputs = []
		self.dset_size = 0
		self.save_hyperparameters()

	def configure_optimizers(self):
		optimizer = torch.optim.AdamW(list(self.embedding.parameters()) + 
									list(self.enc_layers.parameters()) +
									list(self.fc.parameters()),
									lr=0.001)
		return optimizer

	def perform_loss_step(self, batch, mode='train'):
		inputt, labels, ids = batch

		embedded = self.embedding(inputt, ids)

		encoded = embedded
		for enc_layer in self.enc_layers:
			encoded = enc_layer(encoded)
				
		preds_dist = self.fc(encoded)

		# preds_dist_single_marker = preds_dist[torch.arange(inputt.shape[0]), ids, :]
		# preds_single_marker = torch.argmax(preds_dist_single_marker, dim=-1)		
		# correct = torch.sum(preds_single_marker == labels)

		expected_dist = F.one_hot(inputt, num_classes=6).float()
		loss = torch.nn.CrossEntropyLoss()(preds_dist, expected_dist)

		self.log(mode+'_loss', loss)

		# # STATS
		# self.correct_outputs.append(correct)
		# self.dset_size += inputt.shape[0]
		# # STATS

		return loss

	def training_step(self, batch, batch_idx):
		return self.perform_loss_step(batch, mode='train')

	def validation_step(self, batch, batch_idx):
		return self.perform_loss_step(batch, mode='val')
	
	def on_train_epoch_end(self):
		# # do something with all training_step outputs, for example:
		# epoch_mean = torch.stack(self.correct_outputs).sum() / self.dset_size
		# self.log("training_epoch_acc", epoch_mean)
		# # free up the memory
		# self.correct_outputs.clear()
		# self.dset_size = 0
		pass

	def predict(self, x):
		# pass "x" in batch
		inputt, labels, ids = x

		embedded = self.embedding(inputt, ids)

		encoded = embedded
		for enc_layer in self.enc_layers:
				encoded = enc_layer(encoded)
				
		preds_dist = self.fc(encoded)
		preds_dist = preds_dist[torch.arange(inputt.shape[0]), ids, :]
				
		return preds_dist