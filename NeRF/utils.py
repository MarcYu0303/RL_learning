import torch
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
from NeRF_from_scratch import NeRFModel


class NeRF_Evaluation:
    def __init__(self, data_dir='/home/yuran/Data/NeRF',
                 device=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')) -> None:
        self.training_dataset = torch.from_numpy(np.load(f'{data_dir}/training_data.pkl', allow_pickle=True))
        self.testing_dataset = torch.from_numpy(np.load(f'{data_dir}/testing_data.pkl', allow_pickle=True))
        self.device = device
        self.model = NeRFModel(hidden_dim=256).to(device).load_model()
        
    def display_testing_data(self, img_index=0, H=400, W=400):
        ground_truth_pixels = self.testing_dataset[img_index * H * W: (img_index + 1) * H * W, 6:]
        img = ground_truth_pixels.reshape(H, W, 3)
        
        plt.figure()
        plt.imshow(img)
        plt.savefig(f'./NeRF/novel_views/groundtruth_img_{img_index}.png', bbox_inches='tight')
        plt.close()    
        



if __name__ == '__main__':
    evalution = NeRF_Evaluation()
    evalution.display_testing_data()