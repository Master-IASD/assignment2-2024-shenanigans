import argparse
import csv
import datetime
import json
from tqdm import trange
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from model import Generator, Discriminator, initialize_weights
from utils import save_models, gradient_penalty

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser(description='Train Normalizing Flow.')
    parser.add_argument("--z_dim", type=int, default=100, 
                        help="Dimension of the noise vector.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.0002,
                      help="The learning rate to use for training.")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Size of mini-batches for SGD")
    parser.add_argument("--custom_suffix", type=str, default='', 
                        help="Add a custom suffix to the experiment name.")
    
    # WGAN
    critic_iterations = 5
    weight_clip = 0.01
    # WGAN-GP
    # critic_iterations = 5
    # landa = 10

    args = parser.parse_args()
    
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    ts = datetime.datetime.now().isoformat(timespec='minutes')
    ts = ts.replace(':', '-').replace('-', '')
    exp_name = f'exp_{ts}_z{args.z_dim}_lr{args.lr}_epochs{args.epochs}_bs{args.batch_size}'
    if args.custom_suffix != '':
        exp_name += '_' + args.custom_suffix

    os.makedirs(f'checkpoints/{exp_name}/', exist_ok=True)

    # Data Pipeline
    print('Dataset loading...')
    # MNIST Dataset
    transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5), std=(0.5))])

    train_dataset = datasets.MNIST(
        root='data/MNIST/',
        train=True,
        transform=transforms,
        download=True
    )
    test_dataset = datasets.MNIST(
        root='data/MNIST/',
        train=False,
        transform=transforms,
        download=False
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=args.batch_size,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    print('Dataset Loaded.')

    # Model Pipeline
    print('Model Loading...')
    mnist_dim = 784
    G = nn.DataParallel(Generator(z_dim=args.z_dim, g_output_dim=mnist_dim)).to(device)
    # WGAN
    D = nn.DataParallel(Discriminator()).to(device)
    # GAN
    # D = nn.DataParallel(Discriminator(d_input_dim=mnist_dim)).to(device)
    initialize_weights(G)
    initialize_weights(D)

    print('Model loaded.')

    # Optimizers
    # WGAN-GP
    # G_optimizer = optim.Adam(G.parameters(), lr=args.lr, betas=(0.0, 0.9))
    # D_optimizer = optim.Adam(D.parameters(), lr=args.lr, betas=(0.0, 0.9))
    # WGAN
    G_optimizer = optim.RMSprop(G.parameters(), lr=args.lr)
    D_optimizer = optim.RMSprop(D.parameters(), lr=args.lr)
    # GAN
    # G_optimizer = optim.Adam(G.parameters(), lr=args.lr)
    # D_optimizer = optim.Adam(D.parameters(), lr=args.lr)

    # Loss
    # criterion = nn.BCELoss()

    losses = [] # logging

    print('Start Training :')
    G.train()
    D.train()

    pbar = trange(args.epochs)
    for epoch in pbar:
        running_lossD = 0.0
        running_lossG = 0.0

        for batch_i, (real, _) in enumerate(train_loader):
            # WGAN
            real = real.to(device)
            # GAN
            # real = real.view(-1, mnist_dim).to(device)
            
            batch_size = real.shape[0]

            # WGAN
            # Train Discriminator
            for _ in range(critic_iterations):
                noise = torch.randn(batch_size, args.z_dim).to(device)
                fake = G(noise)
                D_real = D(real).reshape(-1)
                D_fake = D(fake.detach()).reshape(-1)

                # WGAN-GP
                # gp = gradient_penalty(D, real, fake, device=device)
                # lossD = -(torch.mean(D_real) - torch.mean(D_fake)) + landa*gp
                # WGAN
                lossD = -(torch.mean(D_real) - torch.mean(D_fake))
                D.zero_grad()
                lossD.backward(retain_graph=True)
                D_optimizer.step()

                running_lossD += lossD.item()

                for p in D.parameters():
                    p.data.clamp_(-weight_clip, weight_clip)

            # Train Generator
            output = D(fake).reshape(-1)
            lossG = -torch.mean(output)

            G.zero_grad()
            lossG.backward()
            G_optimizer.step()

            running_lossG += lossG.item()

            # GAN
            # # Train Discriminator
            # # Train Discriminator on Real
            # D_real = D(real).view(-1)
            # lossD_real = criterion(D_real, torch.ones_like(D_real))

            # # Train Discriminator on Fake
            # noise = torch.randn(batch_size, args.z_dim).to(device)
            # fake = G(noise)
            # D_fake = D(fake.detach()).view(-1)
            # lossD_fake = criterion(D_fake, torch.zeros_like(D_fake))

            # lossD = (lossD_real + lossD_fake) # Maybe divide by 2?
            # D.zero_grad()
            # lossD.backward()
            # D_optimizer.step()

            # running_lossD += lossD.item()

            # # Train Generator
            # output = D(fake).view(-1)
            # lossG = criterion(output, torch.ones_like(output))
            # G.zero_grad()
            # lossG.backward()
            # G_optimizer.step()

            # running_lossG += lossG.item()

        avg_lossD = running_lossD / len(train_loader)
        avg_lossG = running_lossG / len(train_loader)

        losses.append(
            {
                'epoch': epoch,
                'lossD': avg_lossD,
                'lossG': avg_lossG,
            }
        )

        if epoch % 10 == 0:
            save_models(G, D, f'checkpoints/{exp_name}')
                
    print('Training done')

    # Save Runtime
    with open(f'checkpoints/{exp_name}/runtime.json', "w") as file:
        elapsed = pbar.format_dict['elapsed']
        min, sec = divmod(elapsed, 60)
        json.dump({'runtime': f"{int(min):02}:{int(sec):02}"}, file, indent=4)

    # Save Losses
    with open(f'checkpoints/{exp_name}/losses.csv', mode='w', newline='') as file:
        fieldnames = losses[0].keys()
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(losses)

# import torch 
# import os
# from tqdm import trange
# import argparse
# from torchvision import datasets, transforms
# import torch.nn as nn
# import torch.optim as optim

# from model import Generator, Discriminator
# from utils import D_train, G_train, save_models


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Train Normalizing Flow.')
#     parser.add_argument("--epochs", type=int, default=100,
#                         help="Number of epochs for training.")
#     parser.add_argument("--lr", type=float, default=0.0002,
#                       help="The learning rate to use for training.")
#     parser.add_argument("--batch_size", type=int, default=64, 
#                         help="Size of mini-batches for SGD")

#     args = parser.parse_args()


#     os.makedirs('checkpoints', exist_ok=True)
#     os.makedirs('data', exist_ok=True)

#     # Data Pipeline
#     print('Dataset loading...')
#     # MNIST Dataset
#     transform = transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=(0.5), std=(0.5))])

#     train_dataset = datasets.MNIST(root='data/MNIST/', train=True, transform=transform, download=True)
#     test_dataset = datasets.MNIST(root='data/MNIST/', train=False, transform=transform, download=False)


#     train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
#                                                batch_size=args.batch_size, shuffle=True)
#     test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
#                                               batch_size=args.batch_size, shuffle=False)
#     print('Dataset Loaded.')


#     print('Model Loading...')
#     mnist_dim = 784
#     G = torch.nn.DataParallel(Generator(g_output_dim = mnist_dim)).cuda()
#     D = torch.nn.DataParallel(Discriminator(mnist_dim)).cuda()


#     # model = DataParallel(model).cuda()
#     print('Model loaded.')

#     # define loss
#     criterion = nn.BCELoss() 

#     # define optimizers
#     G_optimizer = optim.Adam(G.parameters(), lr = args.lr)
#     D_optimizer = optim.Adam(D.parameters(), lr = args.lr)

#     print('Start Training :')
    
#     n_epoch = args.epochs
#     for epoch in trange(1, n_epoch+1, leave=True):           
#         for batch_idx, (x, _) in enumerate(train_loader):
#             x = x.view(-1, mnist_dim)
#             D_train(x, G, D, D_optimizer, criterion)
#             G_train(x, G, D, G_optimizer, criterion)

#         if epoch % 10 == 0:
#             save_models(G, D, 'checkpoints')
                
#     print('Training done')