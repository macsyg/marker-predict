import torch


x = torch.tensor([[[1,2,3],[4,5,6],[7,8,9]],[[1,2,3],[4,5,6],[7,8,9]],[[1,2,3],[4,5,6],[7,8,9]]])

# z = torch.tensor([[0,1],[1,2], [0,2]])

# print(x[:, z, :])

# print(x)


x = torch.zeros(3,3,3)

# x = torch.tensor([[0,1,2,0,1,2,0,1,2],[0,1,2,0,1,2,0,1,2]])
y = torch.tensor([[1,2]])

z = x[:,y,:]
print(z.shape)