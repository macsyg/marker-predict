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

bins_test = bins_dset[-1000:]
marker_pos_test = markers_dset[-1000:]

test_dset = MarkerDataset(bins_test, marker_pos_test)
test_dataloader = DataLoader(test_dset, batch_size=64, shuffle=True)

# model = Model(dic_size=40, num_bins=len(CELLTYPES), d_embed=128, d_ff=256, num_heads=4, num_layers=8)
model = Model.load_from_checkpoint(checkpoint_path="./lightning_logs/version_3/checkpoints/epoch=499-step=635500.ckpt")

top1_correct = 0
top3_correct = 0
for batch in test_dataloader:

	bins, markers = batch
	missing_ids = torch.randint(0, bins.shape[1], (bins.shape[0],))
	labels = bins[torch.arange(bins.shape[0]), missing_ids] 
	bins[torch.arange(bins.shape[0]), missing_ids] = NUM_BINS

	preds_dist, _ = model.predict((bins, markers, missing_ids))
	
	# checking top1
	preds = torch.argmax(preds_dist, dim=-1)
	top1_correct += torch.sum(torch.eq(labels, preds))

	# checking top3
	_, tk = torch.topk(preds_dist, 3, dim=-1)
	lk = torch.repeat_interleave(labels, 3).reshape(labels.shape[0], 3)
	correct_top3_preds = torch.eq(lk, tk).any(dim=1)
	top3_correct += torch.sum(correct_top3_preds)
	
print("TOP1:", top1_correct / 1000)
print("TOP3:", top3_correct / 1000)