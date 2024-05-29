import torch


xd = torch.Tensor([[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0]])

xd1 = xd.repeat(2, 1, 1)
xd2 = xd.repeat(2, 1, 1)
l = [xd1,xd2]
print(xd1.shape)

xdd = torch.cat(l)
print(xdd.shape)