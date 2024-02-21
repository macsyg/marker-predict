
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

# %%
start_dset = torch.tensor(cell_df[PANEL_1_MARKER_NAMES].values)
start_dset = start_dset.type(torch.LongTensor)

mask = torch.randint(0, 40, (start_dset.shape[0],))
# mask = torch.Tensor([39]).repeat((start_dset.shape[0],)).long()

results = start_dset[torch.arange(start_dset.shape[0]), mask]
# masked_dset = start_dset
# masked_dset[torch.arange(start_dset.shape[0]), mask] = torch.full((start_dset.shape[0],), -1.0, dtype=torch.float64)

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
	

inputs_train = start_dset[0:-1000]
labels_train = results[0:-1000]
ids_train = mask[0:-1000]

inputs_val = start_dset[-1000:]
labels_val = results[-1000:]
ids_val = mask[-1000:]

train_dset = MarkerDataset(inputs_train, labels_train, ids_train)
val_dset = MarkerDataset(inputs_val, labels_val, ids_val)

print(train_dset[0])

train_dataloader = DataLoader(train_dset, batch_size=256, shuffle=True, num_workers=64)
val_dataloader = DataLoader(val_dset, batch_size=64, shuffle=True)

# %%
attn_mask = torch.reshape(torch.Tensor(MISSING_IN_PANEL_2).repeat(len(MISSING_IN_PANEL_2)), (len(MISSING_IN_PANEL_2), len(MISSING_IN_PANEL_2))).bool()

# print(attn_mask)

model = Model(dic_size=40, num_bins=6, d_embed=128, d_ff=256, num_heads=4, num_layers=4, attn_mask=attn_mask)

MAX_EPOCHS = 10

trainer = pl.Trainer(
	max_epochs=MAX_EPOCHS,
	accelerator="gpu",
	# strategy='ddp_find_unused_parameters_true',
	devices=[1, 2, 3]
)

trainer.fit(model, train_dataloader)

trainer.validate(dataloaders=val_dataloader)






