import torch


w = torch.tensor([[[1,2,3],[4,5,6],[7,8,9]],[[1,2,3],[4,5,6],[7,8,9]],[[1,2,3],[4,5,6],[7,8,9]]])

# z = torch.tensor([[0,1],[1,2], [0,2]])

# print(x[:, z, :])

# print(x)


# x = torch.zeros(3,3,3)

# x = torch.tensor([[0,1,2,0,1,2,0,1,2],[0,1,2,0,1,2,0,1,2]])
x = torch.tensor([0, 0, 1, 1, 2, 2])
y = torch.tensor([0, 1, 0, 2, 1, 2])
z = torch.tensor([1, 1, 1, 1, 1, 1])

# z = x[:,y,:]
# print(z.shape)
# print(z)
w[x,y,z] = 0
print(w)