import torch

a = torch.FloatTensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
ii = torch.LongTensor([0, 0, 0, 1, 1, 1, 2, 2, 2])

bla = torch.zeros(3).scatter_add_(0, ii, a)
print(bla)
