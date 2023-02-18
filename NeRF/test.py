import numpy as np
import torch


if __name__ == '__main__':
    a = torch.cat((torch.tensor([0]), torch.tensor([1, 2])))
    a = a.unsqueeze(2)
    print(a)
    print(a.shape)