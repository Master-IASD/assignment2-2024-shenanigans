# import torch
import torch.nn as nn
# import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super().__init__()

        max_dim = 1024

        self.disc = nn.Sequential(
            nn.Linear(d_input_dim, max_dim), # d_input_dim -> 1024
            nn.LeakyReLU(0.2),
            nn.Linear(max_dim, max_dim//2), # 1024 -> 512
            nn.LeakyReLU(0.2),
            nn.Linear(max_dim//2, max_dim//4), # 512 -> 256
            nn.LeakyReLU(0.2),
            nn.Linear(max_dim//4, 1), # 256 -> 1
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim, g_output_dim):
        super().__init__()

        max_dim = 1024

        self.gen = nn.Sequential(
            nn.Linear(z_dim, max_dim//4),
            nn.LeakyReLU(0.2),
            nn.Linear(max_dim//4, max_dim//2),
            nn.LeakyReLU(0.2),
            nn.Linear(max_dim//2, max_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(max_dim, g_output_dim),
            nn.Tanh(),
        )
    
    def forward(self, x):
        return self.gen(x)


# class Generator(nn.Module):
#     def __init__(self, g_output_dim):
#         super(Generator, self).__init__()       
#         self.fc1 = nn.Linear(100, 256)
#         self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
#         self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
#         self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

#     # forward method
#     def forward(self, x): 
#         x = F.leaky_relu(self.fc1(x), 0.2)
#         x = F.leaky_relu(self.fc2(x), 0.2)
#         x = F.leaky_relu(self.fc3(x), 0.2)
#         return torch.tanh(self.fc4(x))

# class Discriminator(nn.Module):
#     def __init__(self, d_input_dim):
#         super(Discriminator, self).__init__()
#         self.fc1 = nn.Linear(d_input_dim, 1024)
#         self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
#         self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
#         self.fc4 = nn.Linear(self.fc3.out_features, 1)

#     # forward method
#     def forward(self, x):
#         x = F.leaky_relu(self.fc1(x), 0.2)
#         x = F.leaky_relu(self.fc2(x), 0.2)
#         x = F.leaky_relu(self.fc3(x), 0.2)
#         return torch.sigmoid(self.fc4(x))
