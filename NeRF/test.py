import numpy as np
import torch


if __name__ == '__main__':
    x = torch.linspace(0, 5, 5)
    print(x, x.shape)
 
    y=x.expand(3, 5)
 
    print(y, y.shape)