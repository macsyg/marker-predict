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

from guess_celltype import Model

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

inputs_val = x[-1000:]
labels_val = y[-1000:]

test_dset = MarkerDataset(inputs_val, labels_val)

test_dataloader = DataLoader(test_dset, batch_size=64)

# model = Model(dic_size=40, num_bins=len(CELLTYPES), d_embed=128, d_ff=256, num_heads=4, num_layers=8)
model = Model.load_from_checkpoint(checkpoint_path="./lightning_logs/version_1/checkpoints/epoch=0-step=3813.ckpt")

print(model.num_bins)

preds_l = []
for batch in test_dataloader:
	inputt, labels = batch
	batch = (inputt.to(DEVICE), labels.to(DEVICE))
	preds = model.predict(batch)
	preds_l.append(preds)

predss = torch.cat(preds_l, 0).to("cpu")
labelss = torch.tensor(labels_val)

cm = confusion_matrix(labelss, predss)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.savefig("./matrices/conf_m1.png")