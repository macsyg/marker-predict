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

CELLTYPES = ['Tumor', 'CD8', 'plasma', 'Mural', 'CD4', 'DC', 'B', 'BnT', 
			 'HLADR', 'MacCD163', 'Neutrophil', 'undefined', 'pDC', 'Treg', 'NK']


def list_to_map(l):
	mapp = {}
	i = 0
	for elem in l:
		mapp[elem] = i
		i += 1
	return mapp

celltype_to_id = list_to_map(CELLTYPES) 

x = torch.tensor(cell_df[PANEL_1_MARKER_NAMES].values)
x = x.type(torch.LongTensor)

y = cell_df['celltypes'].values.tolist()
y = torch.tensor([celltype_to_id[elem] for elem in y])

class MarkerDataset(torch.utils.data.Dataset):
	def __init__(self, x, y):
		super(MarkerDataset, self).__init__()
		# store the raw tensors
		self._x = x
		self._y = y

	def __len__(self):
		# a DataSet must know it size
		return self._x.shape[0]

	def __getitem__(self, index):
		x = self._x[index]
		y = self._y[index]
		return x, y
	

inputs_train = x[0:-1000]
labels_train = y[0:-1000]

inputs_val = x[-1000:]
labels_val = y[-1000:]

train_dset = MarkerDataset(inputs_train, labels_train)
val_dset = MarkerDataset(inputs_val, labels_val)

print(train_dset[20])

train_dataloader = DataLoader(train_dset, batch_size=64, shuffle=True, num_workers=64)
val_dataloader = DataLoader(val_dset, batch_size=64, shuffle=True)


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
	def __init__(self, d_embed, dic_size, num_bins):
		super().__init__()

		self.relu11 =  nn.ReLU()
		self.relu12 =  nn.ReLU()
		self.fc11 = nn.Linear(d_embed, 256)
		self.fc12 = nn.Linear(256, 64)
		self.fc13 = nn.Linear(64, 1)

		self.relu21 =  nn.ReLU()
		self.fc21 = nn.Linear(dic_size, 64)
		self.fc22 = nn.Linear(64, num_bins)
		self.softmax = nn.LogSoftmax(dim=-1)

	def forward(self, x):
		x = self.fc11(x)
		x = self.relu11(x)
		x = self.fc12(x)
		x = self.relu12(x)
		x = self.fc13(x)

		x = torch.squeeze(x)

		x = self.fc21(x)
		x = self.relu21(x)
		x = self.fc22(x)
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
		self.correct_outputs = []
		self.dset_size = 0
		self.save_hyperparameters()

	def configure_optimizers(self):
		optimizer = torch.optim.AdamW(list(self.embedding.parameters()) + 
									list(self.enc_layers.parameters()) +
									list(self.fc.parameters()))
		return optimizer

	def perform_loss_step(self, batch, mode='train'):
		input, labels = batch

		embedded = self.embedding(input)

		encoded = embedded
		for enc_layer in self.enc_layers:
			encoded = enc_layer(encoded)
				
		preds_dist = self.fc(encoded)

		preds = torch.argmax(preds_dist, dim=-1)	
		correct = torch.sum(preds == labels)

		expected_dist = F.one_hot(labels, num_classes=self.num_bins).float()
		loss = torch.nn.CrossEntropyLoss()(preds_dist, expected_dist)

		self.log(mode+'_loss', loss)

		self.correct_outputs.append(correct)
		self.dset_size += input.shape[0]

		return loss, preds_dist

	def training_step(self, batch, batch_idx):
		train_loss, preds_dist = self.perform_loss_step(batch, mode='train')
		# if batch_idx % 100 == 0:
		# 	print(preds_dist[-1])

		return train_loss

	def validation_step(self, batch, batch_idx):
		val_loss, _ = self.perform_loss_step(batch, mode='val')

		return val_loss
	
	def on_train_epoch_end(self):
		# do something with all training_step outputs, for example:
		epoch_mean = torch.stack(self.correct_outputs).sum() / self.dset_size
		self.log("training_epoch_acc", epoch_mean)
		# free up the memory
		self.correct_outputs.clear()
		self.dset_size = 0     
		
model = Model(dic_size=40, num_bins=len(CELLTYPES), d_embed=128, d_ff=256, num_heads=4, num_layers=8)

MAX_EPOCHS = 10

trainer = pl.Trainer(
	max_epochs=MAX_EPOCHS,
	accelerator="gpu",
	# strategy='ddp_find_unused_parameters_true',
	devices=[0, 2, 3]
)

trainer.fit(model, train_dataloader)

trainer.validate(dataloaders=val_dataloader)