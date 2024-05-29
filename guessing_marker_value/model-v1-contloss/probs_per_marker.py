import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import pandas

import matplotlib.pyplot as plt
import numpy as np

import lightning.pytorch as pl

from single_fc_arch import Model

DEVICE = 'cuda'

CELL_DF_PATH = '../../../eb_esb_train_nsclc2_df_bin.df'

cell_df = pandas.read_csv(CELL_DF_PATH)


PANEL_1_MARKER_NAMES = ['MPO', 'HistoneH3', 'SMA', 'CD16', 'CD38',
			 'HLADR', 'CD27', 'CD15', 'CD45RA', 'CD163', 'B2M', 'CD20', 'CD68',
			 'Ido1', 'CD3', 'LAG3', 'CD11c', 'PD1', 'PDGFRb', 'CD7', 'GrzB',
			 'PDL1', 'TCF7', 'CD45RO', 'FOXP3', 'ICOS', 'CD8a', 'CarbonicAnhydrase',
			 'CD33', 'Ki67', 'VISTA', 'CD40', 'CD4', 'CD14', 'Ecad', 'CD303',
			 'CD206', 'cleavedPARP', 'DNA1', 'DNA2']

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

start_dset = torch.tensor(cell_df[PANEL_1_MARKER_NAMES].values)
start_dset = start_dset.type(torch.LongTensor)

mask = torch.randint(0, 39, (start_dset.shape[0],))
results = start_dset[torch.arange(start_dset.shape[0]), mask]

inputs_val = start_dset[-1000:]
labels_val = results[-1000:]
ids_val = mask[-1000:]

model = Model.load_from_checkpoint(checkpoint_path="./lightning_logs/version_5/checkpoints/epoch=19-step=19080.ckpt")

for i in range(40):
	print(i)

	samples = torch.arange(start = start_dset.shape[0]-1000, end = start_dset.shape[0])
	ids_val = (torch.ones(1000) * i).int()

	results = start_dset[samples, ids_val]
	labels_val = results[-1000:]

	test_dset = MarkerDataset(inputs_val, labels_val, ids_val)
	test_dataloader = DataLoader(test_dset, batch_size=64)

	probs_l = []
	for batch in test_dataloader:
		_, labels, _ = batch
		preds_dist = model.predict(batch)
		probs = preds_dist[torch.arange(0, labels.shape[0]),labels]
		probs_l.append(probs)

	log_probss = torch.cat(probs_l, 0)
	probss = torch.exp(log_probss)
	print(probss)

	counts, bins = np.histogram(probss.detach(), range=(0, 1))
	plt.hist(bins[:-1], bins, weights=counts)

	plt.title(PANEL_1_MARKER_NAMES[i] + " prob distrib")
	plt.savefig(f"./histograms/histogram_per_marker/hist{i}.png")
	plt.close()