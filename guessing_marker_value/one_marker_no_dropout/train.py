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

dict_ids = {}
with open(MARKERS_DICT_PATH) as json_file:
    dict_ids = json.load(json_file)

markers_ids = [dict_ids[marker] for marker in PANEL_1_MARKER_NAMES]

# %%
'''
Masking a single token.
'''
# val_dset_size = 1000
# bins_dset = torch.tensor(cell_df[PANEL_1_MARKER_NAMES].values)
# bins_dset = bins_dset.type(torch.LongTensor)
# markers_dset = torch.tensor(markers_ids).repeat(bins_dset.shape[0], 1)

# original_dset_size = bins_dset.shape[0]

# mask = torch.randint(0, len(PANEL_1_MARKER_NAMES), (bins_dset.shape[0],))
# missing_bins = bins_dset[torch.arange(bins_dset.shape[0]), mask]

# bins_train = bins_dset[0:-val_dset_size]
# marker_pos_train = markers_dset[0:-val_dset_size]
# labels_train = missing_bins[0:-val_dset_size]
# missing_pos_train = mask[0:-val_dset_size]

# bins_val = bins_dset[-val_dset_size:]
# marker_pos_val = markers_dset[-val_dset_size:]
# labels_val = missing_bins[-val_dset_size:]
# missing_pos_val = mask[-val_dset_size:]


'''
Masking a single token, but every example in dataset is copied certain amount of times and different tokens are masked.
'''
val_dset_size = 100000
dist_markers_to_mask = 5

bins_dset = torch.tensor(cell_df[PANEL_1_MARKER_NAMES].values)
bins_dset = bins_dset.long()
markers_dset = torch.tensor(markers_ids).repeat(bins_dset.shape[0], 1)

original_dset_size = bins_dset.shape[0]
train_dset_size = original_dset_size - val_dset_size

bins_train = bins_dset[0:-val_dset_size].repeat(dist_markers_to_mask, 1)
marker_pos_train = markers_dset[0:-val_dset_size].repeat(dist_markers_to_mask, 1)
missing_pos_train = torch.stack([torch.randperm(len(PANEL_1_MARKER_NAMES)) for _ in range(train_dset_size)])[:, :dist_markers_to_mask].flatten()
labels_train = bins_train[torch.arange(train_dset_size*dist_markers_to_mask), missing_pos_train]

bins_val = bins_dset[-val_dset_size:]
marker_pos_val = markers_dset[-val_dset_size:]
missing_pos_val = torch.randint(0, len(PANEL_1_MARKER_NAMES), (val_dset_size,))
labels_val = bins_val[torch.arange(val_dset_size), missing_pos_val]


class MarkerDataset(torch.utils.data.Dataset):
	def __init__(self, bins, marker_pos, labels, missing_ids):
		super(MarkerDataset, self).__init__()
		# store the raw tensors
		self._bins = bins
		self._marker_poss = marker_pos
		self._labels = labels
		self._missing_ids = missing_ids

	def __len__(self):
		# a DataSet must know it size
		return self._bins.shape[0]

	def __getitem__(self, index):
		bin_nr = self._bins[index]
		marker_pos = self._marker_poss[index]
		label = self._labels[index]
		missing_id = self._missing_ids[index]
		return bin_nr, marker_pos, label, missing_id
	
# print(bins_train.shape, marker_pos_train.shape, labels_train.shape, missing_pos_train.shape)
# print(bins_val.shape, marker_pos_val.shape, labels_val.shape, missing_pos_val.shape)

train_dset = MarkerDataset(bins_train, marker_pos_train, labels_train, missing_pos_train)
val_dset = MarkerDataset(bins_val, marker_pos_val, labels_val, missing_pos_val)

train_dataloader = DataLoader(train_dset, batch_size=256, shuffle=True, num_workers=64)
val_dataloader = DataLoader(val_dset, batch_size=64, shuffle=True)

model = Model(dic_size=len(dict_ids), num_bins=6, d_embed=128, d_ff=256, num_heads=4, num_layers=4)

MAX_EPOCHS = 10

trainer = pl.Trainer(
	max_epochs=MAX_EPOCHS,
	accelerator="gpu",
	# strategy='ddp_find_unused_parameters_true',
	devices=[1, 2, 3]
)

trainer.fit(model, train_dataloader)

trainer.validate(dataloaders=val_dataloader)
