
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

from common_markers_fc_arch import Model

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

MISSING_IN_PANEL_2 = [True, False, False, True, True,
			True, True, False, False, True, True, False, True,
			True, False, True, False, True, True, False, True,
			True, True, False, False, True, False, True,
			True, False, True, True, True, True, False, True,
			True, True, True, True]

dict_ids = {}
with open(MARKERS_DICT_PATH) as json_file:
    dict_ids = json.load(json_file)

markers_ids = [dict_ids[marker] for marker in PANEL_1_MARKER_NAMES]

unav_markers = torch.squeeze((torch.tensor(MISSING_IN_PANEL_2) == True).nonzero(as_tuple=False))
unav_ids = torch.unsqueeze(torch.tensor(markers_ids), 0)[torch.zeros(unav_markers.shape[0]).long(), unav_markers]
# print(unav_ids)

# %%
bins_dset = torch.tensor(cell_df[PANEL_1_MARKER_NAMES].values)
bins_dset = bins_dset.long()

markers_dset = torch.tensor(markers_ids).repeat(bins_dset.shape[0], 1)

unav_idss_dset = torch.tensor(unav_ids).repeat(bins_dset.shape[0], 1)

class MaskedDifDataset(torch.utils.data.Dataset):
	def __init__(self, bins, marker_pos, unav_idss):
		super(MaskedDifDataset, self).__init__()
		# store the raw tensors
		self._bins = bins
		self._marker_poss = marker_pos
		self._unav_idss = unav_idss

	def __len__(self):
		# a DataSet must know it size
		return self._bins.shape[0]

	def __getitem__(self, index):
		bin_nr = self._bins[index]
		marker_pos = self._marker_poss[index]
		unav_ids = self._unav_idss[index]
		return bin_nr, marker_pos, unav_ids
	

bins_train = bins_dset[0:-1000]
marker_pos_train = markers_dset[0:-1000]
unav_idss_train = unav_idss_dset[0:-1000]

bins_val = bins_dset[-1000:]
marker_pos_val = markers_dset[-1000:]
unav_idss_val = unav_idss_dset[-1000:]

train_dset = MaskedDifDataset(bins_train, marker_pos_train, unav_idss_train)
val_dset = MaskedDifDataset(bins_val, marker_pos_val, unav_idss_val)

train_dataloader = DataLoader(train_dset, batch_size=256, shuffle=True, num_workers=64)
val_dataloader = DataLoader(val_dset, batch_size=64, shuffle=True)

attn_mask = torch.reshape(torch.Tensor(MISSING_IN_PANEL_2).repeat(len(MISSING_IN_PANEL_2)), (len(MISSING_IN_PANEL_2), len(MISSING_IN_PANEL_2))).bool()

model = Model(dic_size=len(dict_ids), num_bins=6, d_embed=128, d_ff=256, num_heads=4, num_layers=4, attn_mask=attn_mask)

MAX_EPOCHS = 30

trainer = pl.Trainer(
	max_epochs=MAX_EPOCHS,
	accelerator="gpu",
	# strategy='ddp_find_unused_parameters_true',
	devices=[0, 1, 2]
)

trainer.fit(model, train_dataloader)

trainer.validate(dataloaders=val_dataloader)






