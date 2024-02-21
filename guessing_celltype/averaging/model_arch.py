import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import lightning.pytorch as pl

def get_positional_encoding(seq_len, embedding_dim):
	v = torch.arange(0, seq_len) * 1.0
	base = (torch.ones(embedding_dim)/10000)
	poww = torch.arange(0, embedding_dim) // 2 * 2.0 / embedding_dim
	denom = torch.pow(base, poww)
	res = torch.matmul(torch.reshape(v, (seq_len,1)), torch.reshape(denom, (1,embedding_dim)))

	res[::2, :] = torch.sin(res[::2, :])
	res[1::2, :] = torch.cos(res[1::2, :])
	positional_encoding = res

	return torch.tensor(positional_encoding, dtype=torch.float)

class EmbedLayer(nn.Module):
	def __init__(self, dic_size=40, embedding_dim=64, num_embeddings=6):
			super().__init__()
			self.embedding_dim = embedding_dim
			self.positional_embedding = nn.Parameter(get_positional_encoding(dic_size, embedding_dim), requires_grad=False)
			# self.name_embedding = nn.Parameter(torch.zeros(size=(dic_size, embedding_dim)))
			self.value_embedding = nn.Embedding(num_embeddings, embedding_dim=embedding_dim)

	def forward(self, x):
			# name_embeds = self.name_embedding.repeat(x.shape[0], 1, 1)
			name_embeds = self.positional_embedding.repeat(x.shape[0], 1, 1)
			value_embeds = self.value_embedding(x)

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
	def __init__(self, d_embed, num_heads, d_ff):
		super().__init__()

		self.to_q = torch.nn.Linear(d_embed, d_embed)
		self.to_k = torch.nn.Linear(d_embed, d_embed)
		self.to_v = torch.nn.Linear(d_embed, d_embed)

		self.attention = torch.nn.MultiheadAttention(embed_dim=d_embed, num_heads=num_heads, batch_first=True)
		self.ff = FeedForward(d_embed, d_ff)
		self.norm1 = nn.LayerNorm(d_embed)
		self.norm2 = nn.LayerNorm(d_embed)

	def forward(self, x):
		q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

		x_tmp1 = x
		x, _ = self.attention(q, k, v)
		x = x + x_tmp1
		x = self.norm1(x)

		x_tmp2 = x
		x = self.ff(x)
		x = x + x_tmp2
		x = self.norm2(x)

		return x

# %%

class Classifier(torch.nn.Module):
	def __init__(self, d_embed, dic_size, num_bins):
		super().__init__()

		self.relu11 =  nn.ReLU()
		self.relu12 =  nn.ReLU()
		self.fc11 = nn.Linear(d_embed, 256)
		self.fc12 = nn.Linear(256, 128)
		self.fc13 = nn.Linear(128, num_bins)
		self.softmax = nn.LogSoftmax(dim=-1)

	def forward(self, x):

		x = torch.mean(x, dim=1)

		x = self.fc11(x)
		x = self.relu11(x)
		x = self.fc12(x)
		x = self.relu12(x)
		x = self.fc13(x)
		x = self.softmax(x)

		return x


# %%
class Model(pl.LightningModule):
	def __init__(self, dic_size, num_bins, d_embed, d_ff, num_heads, num_layers):
		super().__init__()
		self.embedding = EmbedLayer(dic_size=dic_size, embedding_dim=d_embed, num_embeddings=num_bins)
		self.enc_layers = torch.nn.ModuleList([
				EncoderLayer(d_embed, num_heads, d_ff) for i in range(num_layers)
		])
		self.fc = Classifier(d_embed, dic_size, num_bins)
		self.num_bins = num_bins

		# STATS #
		self.correct_train_outputs = 0
		self.correct_val_outputs = 0

		self.train_dset_size = 0
		self.val_dset_size = 0

		self.save_hyperparameters()

	def configure_optimizers(self):
		optimizer = torch.optim.AdamW(list(self.embedding.parameters()) + 
									list(self.enc_layers.parameters()) +
									list(self.fc.parameters()),
									lr=0.0001)
		return optimizer

	def training_step(self, batch, batch_idx):
		inputt, labels = batch

		embedded = self.embedding(inputt)

		encoded = embedded
		for enc_layer in self.enc_layers:
			encoded = enc_layer(encoded)
				
		preds_dist = self.fc(encoded)

		preds = torch.argmax(preds_dist, dim=-1)
		correct = torch.sum(preds == labels)

		expected_dist = F.one_hot(labels, num_classes=self.num_bins).float()
		loss = torch.nn.CrossEntropyLoss()(preds_dist, expected_dist)

		self.log('train_loss', loss)
	
		self.correct_train_outputs += correct
		self.train_dset_size += inputt.shape[0]

		return loss

	def on_train_epoch_end(self):
		# do something with all training_step outputs, for example:
		epoch_mean = self.correct_train_outputs / self.train_dset_size
		self.log("training_epoch_acc", epoch_mean)
		# free up the memory
		self.correct_train_outputs = 0
		self.train_dset_size = 0  

	def validation_step(self, batch, batch_idx):
		inputt, labels = batch

		embedded = self.embedding(inputt)

		encoded = embedded
		for enc_layer in self.enc_layers:
			encoded = enc_layer(encoded)
				
		preds_dist = self.fc(encoded)

		preds = torch.argmax(preds_dist, dim=-1)
		correct = torch.sum(preds == labels)

		expected_dist = F.one_hot(labels, num_classes=self.num_bins).float()
		loss = torch.nn.CrossEntropyLoss()(preds_dist, expected_dist)

		self.correct_val_outputs += correct
		self.val_dset_size += inputt.shape[0]

		return loss

	def on_validation_epoch_end(self):
		# do something with all training_step outputs, for example:
		epoch_mean = self.correct_val_outputs / self.val_dset_size
		self.log("validation_epoch_acc", epoch_mean)
		# free up the memory
		self.correct_val_outputs = 0
		self.val_dset_size = 0  

	def predict(self, x):
		# pass "x" in batch
		inputt, labels = x

		embedded = self.embedding(inputt)

		encoded = embedded
		for enc_layer in self.enc_layers:
				encoded = enc_layer(encoded)
				
		preds_dist = self.fc(encoded)
		preds = torch.argmax(preds_dist, dim=-1)
				  
		return preds