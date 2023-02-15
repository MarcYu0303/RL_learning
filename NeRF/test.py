import numpy as np
import torch


if __name__ == '__main__':
    a = torch.tensor([[1,2], [3,4]])
    c = torch.cumprod(a, dim=1)
    
    print(a[0])
    print(c)