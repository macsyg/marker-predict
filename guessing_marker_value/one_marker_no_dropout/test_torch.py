import torch.nn as nn
import torch

# def compute_joint_attention(att_mat, add_residual=True):
#     if add_residual:
#         residual_att = torch.eye(att_mat.shape[1])[None,...]
#         aug_att_mat = att_mat + residual_att
#         aug_att_mat = aug_att_mat / aug_att_mat.sum(axis=-1)[...,None]
#     else:
#        aug_att_mat =  att_mat
    
#     joint_attentions = torch.zeros(aug_att_mat.shape)

#     layers = joint_attentions.shape[0]
#     joint_attentions[0] = aug_att_mat[0]
#     for i in torch.arange(1,layers):
#         joint_attentions[i] = aug_att_mat[i].dot(joint_attentions[i-1])
        
#     return joint_attentions


# xd = torch.tensor([[1,2],[4,5]])
# xdd = compute_joint_attention(xd)
# print(xdd)

# xd = torch.tensor([[1,2],[4,5]])
# xdd = xd.repeat(2, 1)
# print(xdd)
b, c = 1000000, 40
xd = torch.stack([torch.randperm(c) for _ in range(b)])
print(xd[:, :10].shape)
