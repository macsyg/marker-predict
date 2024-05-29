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

from single_fc_arch import Model

import json

DEVICE = 'cuda'

CELL_DF_PATH = '../../../eb_esb_train_nsclc2_df_bin.df'
MARKERS_DICT_PATH = '../../markers.json'

cell_df = pandas.read_csv(CELL_DF_PATH)


PANEL_1_MARKER_NAMES = ['MPO', 'HistoneH3', 'SMA', 'CD16', 'CD38',
			 'HLADR', 'CD27', 'CD15', 'CD45RA', 'CD163', 'B2M', 'CD20', 'CD68',
			 'Ido1', 'CD3', 'LAG3', 'CD11c', 'PD1', 'PDGFRb', 'CD7', 'GrzB',
			 'PDL1', 'TCF7', 'CD45RO', 'FOXP3', 'ICOS', 'CD8a', 'CarbonicAnhydrase',
			 'CD33', 'Ki67', 'VISTA', 'CD40', 'CD4', 'CD14', 'Ecad', 'CD303',
			 'CD206', 'cleavedPARP', 'DNA1', 'DNA2']

NUM_BINS = 6

dict_ids = {}
with open(MARKERS_DICT_PATH) as json_file:
    dict_ids = json.load(json_file)

markers_ids = [dict_ids[marker] for marker in PANEL_1_MARKER_NAMES]

bins_dset = torch.tensor(cell_df[PANEL_1_MARKER_NAMES].values)
bins_dset = bins_dset.type(torch.LongTensor)
markers_dset = torch.tensor(markers_ids).repeat(bins_dset.shape[0], 1)

class MarkerDataset(torch.utils.data.Dataset):
	def __init__(self, bins, marker_pos):
		super(MarkerDataset, self).__init__()
		self._bins = bins
		self._marker_poss = marker_pos

	def __len__(self):
		return self._bins.shape[0]

	def __getitem__(self, index):
		bin_nr = self._bins[index]
		marker_pos = self._marker_poss[index]
		return bin_nr, marker_pos
	
# print(input_bins[0])

bins_train = bins_dset[0:-1000]
marker_pos_train = markers_dset[0:-1000]

# print(bins_train[0], marker_pos_train[0])

bins_val = bins_dset[-1000:]
marker_pos_val = markers_dset[-1000:]

train_dset = MarkerDataset(bins_train, marker_pos_train)
val_dset = MarkerDataset(bins_val, marker_pos_val)

train_dataloader = DataLoader(train_dset, batch_size=256, shuffle=True, num_workers=64)
val_dataloader = DataLoader(val_dset, batch_size=64, shuffle=True)

model = Model(dic_size=len(dict_ids), 
			  seq_size=len(dict_ids), 
			  num_bins=NUM_BINS+1, 
			  d_embed=256, 
			  d_ff=256, 
			  num_heads=4, 
			  num_layers=4,
			  attn_dropout=0.2,
			  masking="multi")

MAX_EPOCHS = 500

trainer = pl.Trainer(
	max_epochs=MAX_EPOCHS,
	accelerator="gpu",
	# strategy='ddp_find_unused_parameters_true',
	devices=[0, 2, 3]
)

trainer.fit(model, train_dataloader)

trainer.validate(dataloaders=val_dataloader)
