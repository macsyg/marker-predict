import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import pandas

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

import lightning.pytorch as pl

from single_fc_arch import Model

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

dict_ids = {}
with open('markers.json') as json_file:
    dict_ids = json.load(json_file)

markers_ids = [dict_ids[marker] for marker in PANEL_1_MARKER_NAMES]

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

bins_dset = torch.tensor(cell_df[PANEL_1_MARKER_NAMES].values)
bins_dset = bins_dset.long()
markers_dset = torch.tensor(markers_ids).repeat(bins_dset.shape[0], 1)

bins_test = bins_dset[-1000:]
marker_pos_test = markers_dset[-1000:]


model = Model.load_from_checkpoint(checkpoint_path="./lightning_logs/version_25/checkpoints/epoch=9-step=57100.ckpt")

for i in range(40):
	print(i)

	samples = torch.arange(start = bins_dset.shape[0]-1000, end = bins_dset.shape[0])
	missing_pos_test = (torch.ones(1000) * i).int()
	labels_test = bins_dset[samples, missing_pos_test]

	test_dset = MarkerDataset(bins_test, marker_pos_test, labels_test, missing_pos_test)
	test_dataloader = DataLoader(test_dset, batch_size=64)

	preds_l = []
	for batch in test_dataloader:
		preds_dist, _ = model.predict(batch)
		preds = torch.argmax(preds_dist, dim=-1)
		preds_l.append(preds)

	predss = torch.cat(preds_l, 0)

	cm = confusion_matrix(labels_test, predss)

	disp = ConfusionMatrixDisplay(confusion_matrix=cm)
	disp.plot()

	plt.title(PANEL_1_MARKER_NAMES[i])
	plt.savefig(f"./matrices/matrix_per_marker_v25/conf_m{i}.png")
	plt.close()