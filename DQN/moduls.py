import torch

class MPL(torch.nn.Module):

    def __init__(self, obs_size, n_acts):
        super().__init__()
        # self.mlp = self.__mlp(obs_size, n_acts)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(obs_size, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, n_acts),
        )

    def forward(self, x):
        return self.model(x)