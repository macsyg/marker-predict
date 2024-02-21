import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import pandas

import lightning.pytorch as pl

from model_arch import Model

DEVICE = 'cuda'

CELL_DF_PATH = '../../../eb_esb_train_nsclc2_df_bin.df'

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

train_dataloader = DataLoader(train_dset, batch_size=256, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dset, batch_size=64, shuffle=True)

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