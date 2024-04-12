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

	def forward(self, bins, markers, missing_ids):
			name_embeds = self.marker_embedding(markers)
			print(bins)
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
	def __init__(self, d_embed, num_heads, d_ff, attn_mask=None):
		super().__init__()

		self.to_q = torch.nn.Linear(d_embed, d_embed)
		self.to_k = torch.nn.Linear(d_embed, d_embed)
		self.to_v = torch.nn.Linear(d_embed, d_embed)

		self.attention = torch.nn.MultiheadAttention(embed_dim=d_embed, num_heads=num_heads, batch_first=True)
		self.ff = FeedForward(d_embed, d_ff)
		self.norm1 = nn.LayerNorm(d_embed)
		self.norm2 = nn.LayerNorm(d_embed)

		self.attn_mask = nn.Parameter(attn_mask, requires_grad=False) if attn_mask else None

	def forward(self, x):
		q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

		x_tmp1 = x
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
		self.relu = nn.ReLU()
		self.fc1 = nn.Linear(d_embed, 256)
		self.fc2 = nn.Linear(256, 256)
		self.fc3 = nn.Linear(256, num_bins)
		# self.softmax = nn.Softmax(dim=-1)

	def forward(self, x):
		x = self.fc1(x)
		x = self.fc2(self.relu(x))
		x = self.fc3(self.relu(x))
		return x

# %%
class Model(pl.LightningModule):
	def __init__(self, dic_size, num_bins, d_embed, d_ff, num_heads, num_layers, attn_mask=None):
		super().__init__()
		self.embedding = EmbedLayer(dic_size=dic_size, embedding_dim=d_embed, num_embeddings=num_bins)
		self.enc_layers = torch.nn.ModuleList([
				EncoderLayer(d_embed, num_heads, d_ff, attn_mask=attn_mask) for i in range(num_layers)
		])
		self.fc = Classifier(d_embed, num_bins)

		self.num_bins = num_bins
		# self.amt_of_preds_in_epoch = []
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
		bins, markers, labels, missing_ids = batch

		embedded = self.embedding(bins, markers, missing_ids)

		encoded = embedded
		for enc_layer in self.enc_layers:
			encoded = enc_layer(encoded)

		logits = self.fc(encoded)

		logits = logits[torch.arange(bins.shape[0]), missing_ids, :]
		print(logits)
		preds = torch.argmax(logits, dim=-1)	

		# amt_of_preds = torch.bincount(preds, minlength=7)
		correct = torch.sum(preds == labels)

		loss = torch.nn.CrossEntropyLoss()(logits, labels)

		self.log(mode+'_loss', loss)

		self.correct_outputs.append(correct)
		self.dset_size += bins.shape[0]
		# self.amt_of_preds_in_epoch.append(amt_of_preds)

		return loss

	def training_step(self, batch, batch_idx):
		return self.perform_loss_step(batch, mode='train')

	def validation_step(self, batch, batch_idx):
		return self.perform_loss_step(batch, mode='val')
	
	def on_train_epoch_end(self):
		# do something with all training_step outputs, for example:
		correct_classified = torch.stack(self.correct_outputs).sum()
		epoch_mean = correct_classified / self.dset_size

		# amt_of_preds_in_e = torch.stack(self.amt_of_preds_in_epoch).sum(dim=0)

		self.log("good", correct_classified)
		self.log("dset_size", self.dset_size)
		self.log("training_epoch_acc", epoch_mean)
		# free up the memory
		# print(amt_of_preds_in_e)

		# self.amt_of_preds_in_epoch.clear()
		self.correct_outputs.clear()
		self.dset_size = 0

	def predict(self, x):
		# pass "x" in batch
		bins, _, _, missing_ids = x

		embedded = self.embedding(bins, missing_ids)

		encoded = embedded
		for enc_layer in self.enc_layers:
				encoded = enc_layer(encoded)
				
		preds_dist = self.fc(encoded)
		preds_dist = preds_dist[torch.arange(bins.shape[0]), missing_ids, :]
				
		return preds_dist