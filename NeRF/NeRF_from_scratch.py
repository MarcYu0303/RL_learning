import torch
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class NeRFModel(nn.Module):
    def __init__(self, x_embedding_dim=10, d_embedding_dim=4,
                 hidden_dim=256) -> None:
        super(NeRFModel, self).__init__()
        self.block1 = nn.Sequential(
            # +3 --> output = [p]
            # *6 --> for loop: 2(sin,cos) * 3(dimensions) 
            nn.Linear(3 + x_embedding_dim * 6, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        
        self.block2 = nn.Sequential( # input == block1_output + embedded_input
            nn.Linear(3 + x_embedding_dim * 6 + hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim + 1), # +1 is volume density output
            # no nn.ReLU here (see Fig.7)
        )
        
        self.block3 = nn.Sequential(
            nn.Linear(3 + d_embedding_dim * 6 + hidden_dim, hidden_dim // 2), 
            nn.ReLU(),)
        
        self.block4 = nn.Sequential( # output is RGB (emitted color)
            nn.Linear(hidden_dim // 2, 3), nn.Sigmoid(),)
        
        self.x_embedding_dim = x_embedding_dim
        self.d_embedding_dim = d_embedding_dim
        self.ReLU = nn.ReLU()
        self.checkpoint_file = './NeRF/model/NeRF_model'
    
    @staticmethod
    def positional_embedding(p, embedding_length):
        """
        p_embedded = (sin(pi*p), cos(pi*p), ... , sin(2^(L-1)*pi*p), cos(2^(L-1)*pi*p))
        input: 
            p (could be coordinate position or viewing direction)
            embedding_length
        output:
            embedding output
        """
        output = [p] #
        for j in range(embedding_length):
            output.append(torch.sin(2**j * p))
            output.append(torch.cos(2**j * p))
        return torch.cat(output, dim=1)
    
    def forward(self, o, d):
        x_embedding = self.positional_embedding(o, self.x_embedding_dim)
        d_embedding = self.positional_embedding(d, self.d_embedding_dim)
        h = self.block1(x_embedding)
        h_and_sigma = self.block2(torch.cat((h, x_embedding), dim=1))
        h, sigma = h_and_sigma[:, :-1], self.ReLU(h_and_sigma[:, -1]) # ReLU ensures volume density positive
        h = self.block3(torch.cat((h, d_embedding), dim=1))
        c = self.block4(h)
        return c, sigma
    
    def save_model(self, dir=None):
        if dir == None:
            dir = self.checkpoint_file
        torch.save(self.state_dict(), dir)
    
    def load_model(self, dir=None):
        if dir == None:
            dir = self.checkpoint_file
        self.load_state_dict(torch.load(dir))
        

def volume_rendering(nerf_model, ray_origins, ray_directions, tn=0, tf=0.5, num_bins=192):
    '''
    input:
        ray_origins: [batch_size, 3]
        ray_directions: [batch_size, 3]
        num_bins: number of sampling points in a ray (192 --> fine network; 64 --> courase)
    Generated variables:
        t: uniform sample from bins
        delta: delta_i = t_i+1 - t_i    
    Ouput:
        generated pixel values
    '''
    device = ray_origins.device
    t = torch.linspace(tn, tf, num_bins, device=device).expand(ray_origins.shape[0], num_bins)
    # exapnd ray_origins.shape[0] (number of rays) times along the 0 dimension
    
    # generate bins upper and lower bounds
    mid = (t[:, :-1] - t[:, 1:]) / 2
    lower = torch.cat((t[:, :1], mid), -1)
    upper = torch.cat((mid, t[:, -1:]), -1)
    u =  torch.rand(t.shape, device=device) # perturb sampling 
    t = lower + (upper - lower) * u # [batch_size, num_bins]
    
    delta = torch.cat((t[:, 1:] - t[:, :-1], torch.tensor([1e10], device=device).expand(ray_origins.shape[0], 1)), -1)
    
    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)   # [batch_size, num_bins, 3]
    ray_directions = ray_directions.expand(num_bins, ray_directions.shape[0], 3).transpose(0, 1)
    
    colors, sigma = nerf_model(x.reshape(-1, 3), ray_directions.reshape(-1, 3))
    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])
    
    alpha = 1 - torch.exp(-sigma * delta)  # [batch_size, num_bins]
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2) # 1 - alpha = torch.exp(-sigma * delta)
    c = (weights * colors).sum(dim=1) # pixel values
    weight_sum = weights.sum(-1).sum(-1)  # Regularization for white background
    return c + 1 - weight_sum.unsqueeze(-1)
    
def compute_accumulated_transmittance(alphas):
    '''
    accumulated transmittance: T_i = exp(-1 * sum{sigma_i * delta_i}) 
    '''
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
                      accumulated_transmittance[:, :-1]), dim=-1)

@torch.no_grad()
def test(model, tn, tf, dataset, chunk_size=10, img_index=0, num_bins=192, H=400, W=400):
    ray_origins = dataset[img_index * H * W: (img_index + 1) * H * W, :3]
    ray_directions = dataset[img_index * H * W: (img_index + 1) * H * W, 3:6]

    data = []
    for i in range(int(np.ceil(H / chunk_size))):
        ray_origins_ = ray_origins[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        ray_directions_ = ray_directions[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)

        regenerated_px_values = volume_rendering(model, ray_origins_, ray_directions_, tn=tn, tf=tf, num_bins=num_bins)
        data.append(regenerated_px_values)
    img = torch.cat(data).data.cpu().numpy().reshape(H, W, 3)

    plt.figure()
    plt.imshow(img)
    plt.savefig(f'./NeRF/novel_views/img_{img_index}.png', bbox_inches='tight')
    plt.close()    

def train(nerf_model, optimizer, scheduler, data_loader,  device, 
          num_epochs=int(1e5), tn=0, tf=1, num_bins=192, H=400, W=400) -> None:
    for _ in tqdm(range(num_epochs)):
        training_loss = []
        for batch in tqdm(data_loader, total=int(training_dataset.shape[0]/batch_size)):
            ray_origins = batch[:, :3].to(device)
            ray_directions = batch[:, 3:6].to(device)
            ground_truth_pixels = batch[:, 6:].to(device)
            regenerated_pixels = volume_rendering(nerf_model, ray_origins, ray_directions)            
            
            loss = nn.functional.mse_loss(ground_truth_pixels, regenerated_pixels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss.append(loss.item)
            writer.add_scalar(tag='loss', scalar_value=training_loss)
        scheduler.step() # scheduler is used to change lr after certain epochs
        nerf_model.save_model()
        
        for img_index in range(10):
            test(nerf_model, tn, tf, testing_dataset, img_index=img_index, num_bins=num_bins, H=H, W=W)
       
            
if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(device)
    batch_size = 1024
    time = datetime.now().strftime('%Y_%m_%d_%H_%M')
    
    training_dataset = torch.from_numpy(np.load('/home/yuran/Data/NeRF/training_data.pkl', allow_pickle=True))
    testing_dataset = torch.from_numpy(np.load('/home/yuran/Data/NeRF/testing_data.pkl', allow_pickle=True))
    model = NeRFModel(hidden_dim=256).to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=[2, 4, 8], gamma=0.5)
    writer = SummaryWriter(log_dir=time)

    data_loader = DataLoader(training_dataset, batch_size=1024, shuffle=True)
    train(model, model_optimizer, scheduler, data_loader, num_epochs=16, device=device, tn=2, tf=6, num_bins=192, H=400,
          W=400)