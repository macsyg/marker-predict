import torch


t = torch.tensor([[[1., 2.], [1., 2.]], [[1., 2.], [1., 2.]]])
print(t.shape)

t = torch.mean(t, dim=1)
print(t)
print(t.shape)