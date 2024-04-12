# import matplotlib.pyplot as plt
# import torch
# import torch.nn.functional as F
# import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader

# import pandas

# import lightning.pytorch as pl

# from single_fc_arch import Model

# DEVICE = 'cuda'

# CELL_DF_PATH = '../../../eb_esb_train_nsclc2_df_bin.df'

# cell_df = pandas.read_csv(CELL_DF_PATH)


# PANEL_1_MARKER_NAMES = ['MPO', 'HistoneH3', 'SMA', 'CD16', 'CD38',
# 			 'HLADR', 'CD27', 'CD15', 'CD45RA', 'CD163', 'B2M', 'CD20', 'CD68',
# 			 'Ido1', 'CD3', 'LAG3', 'CD11c', 'PD1', 'PDGFRb', 'CD7', 'GrzB',
# 			 'PDL1', 'TCF7', 'CD45RO', 'FOXP3', 'ICOS', 'CD8a', 'CarbonicAnhydrase',
# 			 'CD33', 'Ki67', 'VISTA', 'CD40', 'CD4', 'CD14', 'Ecad', 'CD303',
# 			 'CD206', 'cleavedPARP', 'DNA1', 'DNA2']

# start_dset = torch.tensor(cell_df[PANEL_1_MARKER_NAMES].values)
# start_dset = start_dset.type(torch.LongTensor)

# mask = torch.randint(0, 39, (start_dset.shape[0],))
# results = start_dset[torch.arange(start_dset.shape[0]), mask]

# class MarkerDataset(torch.utils.data.Dataset):
# 	def __init__(self, x, y, mask):
# 		super(MarkerDataset, self).__init__()
# 		# store the raw tensors
# 		self._x = x
# 		self._y = y
# 		self._mask = mask

# 	def __len__(self):
# 		# a DataSet must know it size
# 		return self._x.shape[0]

# 	def __getitem__(self, index):
# 		x = self._x[index]
# 		y = self._y[index]
# 		mask = self._mask[index]
# 		return x, y, mask

# inputs_val = start_dset[-1000:]
# labels_val = results[-1000:]
# ids_val = mask[-1000:]

# test_dset = MarkerDataset(inputs_val, labels_val, ids_val)
# test_dataloader = DataLoader(test_dset, batch_size=64, shuffle=True)

# # model = Model(dic_size=40, num_bins=len(CELLTYPES), d_embed=128, d_ff=256, num_heads=4, num_layers=8)
# model = Model.load_from_checkpoint(checkpoint_path="./lightning_logs/version_0/checkpoints/epoch=19-step=25420.ckpt")

# top1_correct = 0
# top3_correct = 0
# for batch in test_dataloader:
# 	_, labels, _ = batch
# 	preds_dist = model.predict(batch)
	
# 	# checking top1
# 	preds = torch.argmax(preds_dist, dim=-1)
# 	top1_correct += torch.sum(torch.eq(labels, preds))

# 	# checking top3
# 	_, tk = torch.topk(preds_dist, 3, dim=-1)
# 	lk = torch.repeat_interleave(labels, 3).reshape(labels.shape[0], 3)
# 	correct_top3_preds = torch.eq(lk, tk).any(dim=1)
# 	top3_correct += torch.sum(correct_top3_preds)
	
# print("TOP1:", top1_correct / 1000)
# print("TOP3:", top3_correct / 1000)

import torch
xd = torch.tensor([[[1, 2, 3], [1, 2, 4], [1, 2, 5]], [[1, 2, 3], [1, 2, 4], [1, 2, 5]]]).float()
xd = torch.nn.functional.softmax(xd, dim = 0)
print(xd)