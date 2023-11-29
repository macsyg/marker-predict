
# %%
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import pandas

import lightning.pytorch as pl

DEVICE = 'cuda'

CELL_DF_PATH = '../../../../raid/immucan/immuw/atcel/data/eb_esb_train_nsclc2_df_bin.df'

cell_df = pandas.read_csv(CELL_DF_PATH)


PANEL_1_MARKER_NAMES = ['MPO', 'HistoneH3', 'SMA', 'CD16', 'CD38',
			 'HLADR', 'CD27', 'CD15', 'CD45RA', 'CD163', 'B2M', 'CD20', 'CD68',
			 'Ido1', 'CD3', 'LAG3', 'CD11c', 'PD1', 'PDGFRb', 'CD7', 'GrzB',
			 'PDL1', 'TCF7', 'CD45RO', 'FOXP3', 'ICOS', 'CD8a', 'CarbonicAnhydrase',
			 'CD33', 'Ki67', 'VISTA', 'CD40', 'CD4', 'CD14', 'Ecad', 'CD303',
			 'CD206', 'cleavedPARP', 'DNA1', 'DNA2']

start_dset = torch.tensor(cell_df[PANEL_1_MARKER_NAMES].values)
start_dset = start_dset.type(torch.LongTensor)

mask = torch.randint(0, 39, (start_dset.shape[0],))
# mask = torch.Tensor([39]).repeat((start_dset.shape[0],)).long()

results = start_dset[torch.arange(start_dset.shape[0]), mask]
# masked_dset = start_dset
# masked_dset[torch.arange(start_dset.shape[0]), mask] = torch.full((start_dset.shape[0],), -1.0, dtype=torch.float64)

class MarkerDataset(torch.utils.data.Dataset):
	def __init__(self, x, y, mask):
		super(MarkerDataset, self).__init__()
		# store the raw tensors
		self._x = x
		self._y = y
		self._mask = mask

	def __len__(self):
		# a DataSet must know it size
		return self._x.shape[0]

	def __getitem__(self, index):
		x = self._x[index]
		y = self._y[index]
		mask = self._mask[index]
		return x, y, mask
	

inputs_train = start_dset[0:-1000]
labels_train = results[0:-1000]
ids_train = mask[0:-1000]

inputs_val = start_dset[-1000:]
labels_val = results[-1000:]
ids_val = mask[-1000:]

train_dset = MarkerDataset(inputs_train, labels_train, ids_train)
val_dset = MarkerDataset(inputs_val, labels_val, ids_val)

print(train_dset[0])

train_dataloader = DataLoader(train_dset, batch_size=64, shuffle=True, num_workers=64)
val_dataloader = DataLoader(val_dset, batch_size=64, shuffle=True)

class EmbedLayer(nn.Module):
	def __init__(self, dic_size=40, embedding_dim=64, num_embeddings=6):
			super().__init__()
			self.embedding_dim = embedding_dim
			self.name_embedding = nn.Parameter(torch.zeros(size=(dic_size, embedding_dim)))
			self.value_embedding = nn.Embedding(num_embeddings, embedding_dim=embedding_dim)
			self.mask_embedding = nn.Parameter(torch.rand(size=(embedding_dim,)))

	def forward(self, x, y):
			name_embeds = self.name_embedding.repeat(x.shape[0], 1, 1)
			mask_embeds = self.mask_embedding.repeat(x.shape[0], 1)
			# x[torch.arange(x.shape[0]), y] = 6
			value_embeds = self.value_embedding(x)
			# value_embeds[torch.arange(x.shape[0]), y, :] = torch.zeros(x.shape[0], self.embedding_dim).to(DEVICE)
			value_embeds[torch.arange(x.shape[0]), y, :] = mask_embeds

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
	def __init__(self, d_embed, num_heads, d_ff):
		super().__init__()

		self.to_q = torch.nn.Linear(d_embed, d_embed)
		self.to_k = torch.nn.Linear(d_embed, d_embed)
		self.to_v = torch.nn.Linear(d_embed, d_embed)

		self.attention = torch.nn.MultiheadAttention(embed_dim=d_embed, num_heads=num_heads)
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
	def __init__(self, dic_size, num_bins, d_embed, d_ff, num_heads, num_layers):
		super().__init__()
		self.embedding = EmbedLayer(dic_size=dic_size, embedding_dim=d_embed, num_embeddings=num_bins)
		self.enc_layers = torch.nn.ModuleList([
				EncoderLayer(d_embed, num_heads, d_ff) for i in range(num_layers)
		])
		self.fc = Classifier(d_embed, num_bins)
		self.correct_outputs = []
		self.dset_size = 0
		self.save_hyperparameters()

	def configure_optimizers(self):
		optimizer = torch.optim.AdamW(list(self.embedding.parameters()) + 
																	list(self.enc_layers.parameters()) +
																	list(self.fc.parameters()))
		return optimizer

	def perform_loss_step(self, batch, mode='train'):
		input, labels, ids = batch

		embedded = self.embedding(input, ids)

		encoded = embedded
		for enc_layer in self.enc_layers:
			encoded = enc_layer(encoded)
				
		preds_dist = self.fc(encoded)

		# preds = torch.stack([results[i, ids[i], :] for i in range(ids.shape[0])])
		preds_dist = preds_dist[torch.arange(input.shape[0]), ids, :]
		preds = torch.argmax(preds_dist, dim=-1)		
		correct = torch.sum(preds == labels)

		expected_dist = F.one_hot(labels, num_classes=6).float()
		# loss = F.mse_loss(preds_dist, expected_dist) #TODO: tu ma być cross entropy loss
		loss = torch.nn.CrossEntropyLoss()(preds_dist, expected_dist)
		# print(loss)

		self.log(mode+'_loss', loss)

		self.correct_outputs.append(correct)
		self.dset_size += input.shape[0]

		return loss

	def training_step(self, batch, batch_idx):
		return self.perform_loss_step(batch, mode='train')

	def validation_step(self, batch, batch_idx):
		return self.perform_loss_step(batch, mode='val')
	
	def on_train_epoch_end(self):
		# do something with all training_step outputs, for example:
		epoch_mean = torch.stack(self.correct_outputs).sum() / self.dset_size
		self.log("training_epoch_acc", epoch_mean)
		# free up the memory
		self.correct_outputs.clear()
		self.dset_size = 0

	def predict(self, x):
		# pass "x" in batch
		input, labels, ids = x

		embedded = self.embedding(input, ids)

		encoded = embedded
		for enc_layer in self.enc_layers:
				encoded = enc_layer(encoded)
				
		preds = self.fc(encoded)
		preds = preds[torch.arange(x.shape[0]), ids, :]
		preds = torch.argmax(preds, dim=-1)
				
		results = torch.sum(preds == labels)
		print(results)        
		
model = Model(dic_size=40, num_bins=6, d_embed=128, d_ff=256, num_heads=4, num_layers=4)

MAX_EPOCHS = 20

trainer = pl.Trainer(
	max_epochs=MAX_EPOCHS,
	accelerator="gpu",
	# strategy='ddp_find_unused_parameters_true',
	devices=[0, 1, 3]
)

trainer.fit(model, train_dataloader)

trainer.validate(dataloaders=val_dataloader)






