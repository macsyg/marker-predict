import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.distributions import Categorical

import pandas

import lightning.pytorch as pl

from single_fc_arch import Model

import json

import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics

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

# model = Model(dic_size=40, num_bins=len(CELLTYPES), d_embed=128, d_ff=256, num_heads=4, num_layers=8)
model = Model.load_from_checkpoint(checkpoint_path="./lightning_logs/version_4/checkpoints/epoch=19-step=25420.ckpt")

model.train()
# x=True

top1_correct = 0
all_results_logits = []
labels_from_loader = []
for batch in test_dataloader:
	batch_logits = []
	for i in range(16):
		_, _, labels, _ = batch
		preds_logits = model.predict(batch)
		# print(preds_dist.shape)
		preds_logits = torch.unsqueeze(preds_logits, dim=1)
		# print(preds_dist.shape)
		batch_logits.append(preds_logits)
		if i == 0:
			labels_from_loader.append(labels)
	all_results_logits.append(torch.cat(batch_logits, dim=1))
all_results_logits = torch.cat(all_results_logits, dim=0)
labels_val = torch.cat(labels_from_loader)

print(all_results_logits.shape)

all_results_preds = torch.argmax(all_results_logits, dim=2)
determ_preds = torch.mode(all_results_preds, dim=1).values
is_pred_correct = torch.eq(labels_val, determ_preds)
correct_determ_preds_amt =  torch.sum(is_pred_correct)
print("DETERM_PRED_ACC:", correct_determ_preds_amt/1000)

all_results_probs = torch.softmax(all_results_logits, dim=2)
all_results_avg_probs = torch.mean(all_results_probs, dim=1)
avg_preds = torch.argmax(all_results_avg_probs, dim=1)
correct_avg_preds_amt =  torch.sum(torch.eq(labels_val, avg_preds))
print("AVG_PRED_ACC:", correct_avg_preds_amt/1000)

entropy_of_mean = Categorical(all_results_avg_probs).entropy()
all_results_entropies = Categorical(all_results_probs).entropy()
mean_of_entropies = torch.mean(all_results_entropies, dim=1)

every_example_BALD = entropy_of_mean - mean_of_entropies

ids = torch.arange(0, 1000)

fig = plt.figure(figsize =(10, 7))
bars = plt.bar(ids, entropy_of_mean.detach())
for i in range(1000):
	if is_pred_correct[i]:
		bars[i].set_color("green")
	else:
		bars[i].set_color("red")
plt.savefig('histograms/entropy_of_means.png')
plt.clf()

fig = plt.figure(figsize =(10, 7))
bars = plt.bar(ids, every_example_BALD.detach())
for i in range(1000):
	if is_pred_correct[i]:
		bars[i].set_color("green")
	else:
		bars[i].set_color("red")
plt.savefig('histograms/BALD.png')
plt.clf()

y_true = is_pred_correct.long().detach().numpy()
y_score = entropy_of_mean.detach().numpy()

fpr, tpr, thresholds = metrics.roc_curve(y_true, 
										 y_score)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                estimator_name='marker value correctness estimator')
display.plot()
plt.savefig('histograms/roc_auc_entropy_of_mean.png')
plt.clf()


y_true = is_pred_correct.long().detach().numpy()
y_score = every_example_BALD.detach().numpy()

fpr, tpr, thresholds = metrics.roc_curve(y_true, 
										 y_score)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                estimator_name='marker value correctness estimator')
display.plot()
plt.savefig('histograms/roc_auc_bald.png')
plt.clf()


avg_prec_score = metrics.average_precision_score(y_true, y_score)
print("AVG_PREC_SCORE:", avg_prec_score)



fig = plt.figure(figsize =(10, 7))
hist_correct = plt.hist(every_example_BALD.detach()[is_pred_correct], alpha=0.3, bins=50, density=True)
hist_incorrect = plt.hist(every_example_BALD.detach()[np.logical_not(is_pred_correct.numpy())], alpha=0.3, bins=50, density=True)
plt.savefig('histograms/correct_incorrect.png')
plt.clf()