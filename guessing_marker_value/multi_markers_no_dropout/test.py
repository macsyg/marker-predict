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
with open('markers.json') as json_file:
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

bins_test = bins_dset[-1000:]
marker_pos_test = markers_dset[-1000:]
unav_idss_test = unav_idss_dset[-1000:]

test_dset = MaskedDifDataset(bins_test, marker_pos_test, unav_idss_test)
# test_dataloader = DataLoader(test_dset, batch_size=64, shuffle=True)

model = Model.load_from_checkpoint(checkpoint_path="./lightning_logs/version_2/checkpoints/epoch=29-step=38130.ckpt")
model.eval()


for marker_id in unav_ids:
	top1_correct = 0
	top3_correct = 0
	test_dataloader = DataLoader(test_dset, batch_size=64, shuffle=True)
	for batch in test_dataloader:
		bins, markers, unav_ids = batch

		preds_dist = model.predict(batch)
		preds_dist = preds_dist[:,marker_id,:].squeeze()

		correct_values = bins[:, marker_id]

		# checking top1
		preds = torch.argmax(preds_dist, dim=-1)
		top1_correct += torch.sum((preds == correct_values).int())

		# checking top3
		_, tk = torch.topk(preds_dist, 3, dim=-1)
		lk = torch.repeat_interleave(correct_values, 3).reshape(correct_values.shape[0], 3)
		correct_top3_preds = torch.eq(lk, tk).any(dim=1)
		top3_correct += torch.sum(correct_top3_preds)

	print(f"ACCURACY FOR MARKER {PANEL_1_MARKER_NAMES[marker_id]}")
	print("TOP1:", top1_correct / 1000)
	print("TOP3:", top3_correct / 1000)