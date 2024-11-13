# import torch
import torch.nn as nn
# import torch.nn.functional as F

# class Discriminator(nn.Module):
#     def __init__(self, d_input_dim):
#         super().__init__()

#         max_dim = 1024

#         self.disc = nn.Sequential(
#             nn.Linear(d_input_dim, max_dim), # d_input_dim -> 1024
#             nn.LeakyReLU(0.2),
#             nn.Linear(max_dim, max_dim//2), # 1024 -> 512
#             nn.LeakyReLU(0.2),
#             nn.Linear(max_dim//2, max_dim//4), # 512 -> 256
#             nn.LeakyReLU(0.2),
#             nn.Linear(max_dim//4, 1), # 256 -> 1
#             nn.Sigmoid(),
#         )

#     def forward(self, x):
#         return self.disc(x)

# DCGAN Discriminator
class Discriminator(nn.Module):
    def __init__(self, channels_img=1, features_d=64):
        super().__init__()

        self.disc = nn.Sequential(
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2), # (1, 28, 28) -> (featrues_d, 14, 14)
            self._block(features_d, features_d*2, 4, 2, 1), # (features_d, 14, 14) -> (features_d*2, 7, 7)
            self._block(features_d*2, features_d*4, 3, 2, 1), # (features_d*2, 7, 7) -> (features_d*4, 4, 4)
            nn.Conv2d(features_d*4, 1, 4, 2, 0), # (features_d*4, 4, 4) -> (1, 1, 1)
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            # nn.BatchNorm2d(out_channels),
            # WGAN-GP
            # nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
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
        out = self.gen(x)
        return out.view(out.size(0), 1, 28, 28)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


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
