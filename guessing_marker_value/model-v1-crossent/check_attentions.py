import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import pandas as pd

import lightning.pytorch as pl

from single_fc_arch import Model

import json

import seaborn as sns

DEVICE = 'cuda'

CELL_DF_PATH = '../../../eb_esb_train_nsclc2_df_bin.df'

cell_df = pd.read_csv(CELL_DF_PATH)

PANEL_1_MARKER_NAMES = ['MPO', 'HistoneH3', 'SMA', 'CD16', 'CD38',
			 'HLADR', 'CD27', 'CD15', 'CD45RA', 'CD163', 'B2M', 'CD20', 'CD68',
			 'Ido1', 'CD3', 'LAG3', 'CD11c', 'PD1', 'PDGFRb', 'CD7', 'GrzB',
			 'PDL1', 'TCF7', 'CD45RO', 'FOXP3', 'ICOS', 'CD8a', 'CarbonicAnhydrase',
			 'CD33', 'Ki67', 'VISTA', 'CD40', 'CD4', 'CD14', 'Ecad', 'CD303',
			 'CD206', 'cleavedPARP', 'DNA1', 'DNA2']

def compute_joint_attention(att_mat, residual_lambda=0):
	if residual_lambda > 0:
		residual_att = torch.eye(att_mat.shape[1])
		aug_att_mat = att_mat + residual_att
		aug_att_mat = aug_att_mat / aug_att_mat.sum(axis=-1)[...,None]
	else:
		aug_att_mat =  att_mat
	
	joint_attentions = torch.zeros(aug_att_mat.shape)

	result = aug_att_mat[0]

	layers = joint_attentions.shape[0]
	# joint_attentions[0] = aug_att_mat[0]
	for i in torch.arange(1,layers):
		# joint_attentions[i] = aug_att_mat[i].dot(joint_attentions[i-1])
		result = result @ aug_att_mat[i]
		
	return result



dict_ids = {}
with open('markers.json') as json_file:
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
model = Model.load_from_checkpoint(checkpoint_path="./lightning_logs/version_2/checkpoints/epoch=499-step=635500.ckpt")
model.eval()

# for batch in test_dataloader:
# 	_, labels, _ = batch
# 	predicted_logits, att_weights_all_layers = model.predict(batch)

# 	for att_weights in att_weights_all_layers:
# 		print(att_weights.shape)

batch = next(iter(test_dataloader))
bins, markers = batch
missing_ids = torch.Tensor([21]).repeat(64).long()
print(missing_ids)


predicted_logits, att_weights_all_layers = model.predict((bins, markers, missing_ids))

att_ws = []

for att_weights in att_weights_all_layers:
	print(att_weights.shape)
	att_w = torch.mean(att_weights, dim=1).unsqueeze(dim=1)
	print(att_w.shape)
	# att_ws.append(att_w[:,0,:])
	att_ws.append(att_w)
	# print(att_weights.shape)

att_ws = torch.cat(att_ws, dim=1)
print(att_ws.shape)

joint_atts = []
for i in range(att_ws.shape[0]):
	joint_att = compute_joint_attention(att_ws[i].squeeze())
	joint_atts.append(joint_att)

joint_atts = torch.stack(joint_atts)
joint_atts_for_one_marker = joint_atts[:, missing_ids[0],:]
print(joint_atts_for_one_marker.shape)

df = pd.DataFrame(joint_atts_for_one_marker.detach().numpy(), columns=PANEL_1_MARKER_NAMES)
cl = sns.clustermap(df, xticklabels=PANEL_1_MARKER_NAMES,z_score=0, center=0)
cl.savefig('./clustermaps/pdl1_how_affects.png')

attn_mask = torch.zeros((len(PANEL_1_MARKER_NAMES), len(PANEL_1_MARKER_NAMES)))
attn_mask[28, :] = torch.ones(len(PANEL_1_MARKER_NAMES))


# model.attn_mask = F.one_hot(missing_ids[0], num_classes=len(PANEL_1_MARKER_NAMES)).bool().repeat(len(PANEL_1_MARKER_NAMES), 1) 
model.attn_mask = attn_mask.bool()
torch.set_printoptions(profile="full")
print(model.attn_mask)
torch.set_printoptions(profile="default")
predicted_logits, att_weights_all_layers = model.predict((bins, markers, missing_ids))

att_ws = []

torch.set_printoptions(profile="full")
print(att_weights_all_layers[0])
torch.set_printoptions(profile="default")

for att_weights in att_weights_all_layers:
	print(att_weights.shape)
	att_w = torch.mean(att_weights, dim=1).unsqueeze(dim=1)
	print(att_w.shape)
	# att_ws.append(att_w[:,0,:])
	att_ws.append(att_w)
	# print(att_weights.shape)

att_ws = torch.cat(att_ws, dim=1)
print(att_ws.shape)

joint_atts = []
for i in range(att_ws.shape[0]):
	joint_att = compute_joint_attention(att_ws[i].squeeze())
	joint_atts.append(joint_att)

joint_atts = torch.stack(joint_atts)
joint_atts_for_one_marker = joint_atts[:, missing_ids[0],:]
torch.set_printoptions(profile="full")
print(joint_atts_for_one_marker)
torch.set_printoptions(profile="default")

joint_atts1 = joint_atts
joint_atts_for_one_marker1 = joint_atts1[:, :, missing_ids[0]]
joint_atts_for_one_marker1 = torch.where(joint_atts_for_one_marker1 > -10, joint_atts_for_one_marker1, -10)
df1 = pd.DataFrame(joint_atts_for_one_marker1.detach().numpy(), columns=PANEL_1_MARKER_NAMES)
cl1 = sns.clustermap(df1, xticklabels=PANEL_1_MARKER_NAMES,z_score=0, center=0)
cl1.savefig('./clustermaps/pdl1_who_affects.png')


# iris = sns.load_dataset("iris")
# print(iris)
# species = iris.pop("species")
# cl = sns.clustermap(iris)
# cl.savefig('./clustermaps/a.png')