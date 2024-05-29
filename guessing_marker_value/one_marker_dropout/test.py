import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import pandas
import json

import lightning.pytorch as pl

from single_fc_arch import Model

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
bins_dset = torch.tensor(cell_df[PANEL_1_MARKER_NAMES].values)
bins_dset = bins_dset.type(torch.LongTensor)

markers_dset = torch.tensor(markers_ids).repeat(bins_dset.shape[0], 1)

mask = torch.randint(0, len(PANEL_1_MARKER_NAMES), (bins_dset.shape[0],))
missing_bins = bins_dset[torch.arange(bins_dset.shape[0]), mask]

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
	

bins_train = bins_dset[0:-1000]
marker_pos_train = markers_dset[0:-1000]
labels_train = missing_bins[0:-1000]
missing_pos_train = mask[0:-1000]

bins_val = bins_dset[-1000:]
marker_pos_val = markers_dset[-1000:]
labels_val = missing_bins[-1000:]
missing_pos_val = mask[-1000:]

test_dset = MarkerDataset(bins_val, marker_pos_val, labels_val, missing_pos_val)
test_dataloader = DataLoader(test_dset, batch_size=64, shuffle=True)


model = Model.load_from_checkpoint(checkpoint_path="./lightning_logs/version_0/checkpoints/epoch=19-step=25420.ckpt")

top1_correct = 0
top3_correct = 0
for batch in test_dataloader:
	_, _, labels, _ = batch
	preds_dist = model.predict(batch)
	
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
