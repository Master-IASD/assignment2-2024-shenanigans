import torch 
import torchvision
import argparse
import os
import re

from model import Generator
from utils import load_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Normalizing Flow.')
    parser.add_argument("--batch_size", type=int, default=2048,
                      help="The batch size to use for training.")
    parser.add_argument("--checkpoint_directory", type=str, default='checkpoints',
                      help="The directory where the model is located.")
    args = parser.parse_args()

    if args.checkpoint_directory != 'checkpoints':
        match = re.search(r'z(\d+)', args.checkpoint_directory)

        if match:
            z_dim = int(match.group(1))

    else:
        z_dim = 100

    print('Model Loading...')
    # Model Pipeline
    mnist_dim = 784

    model = Generator(g_output_dim=mnist_dim, z_dim=z_dim).cuda()
    model = load_model(model, args.checkpoint_directory)
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    print('Model loaded.')

    print('Start Generating')
    os.makedirs(f'{args.checkpoint_directory}/samples', exist_ok=True)

    n_samples = 0
    with torch.no_grad():
        while n_samples<10000:
            z = torch.randn(args.batch_size, 100).cuda()
            x = model(z)
            x = x.reshape(args.batch_size, 28, 28)
            for k in range(x.shape[0]):
                if n_samples<10000:
                    torchvision.utils.save_image(x[k:k+1], os.path.join(f'{args.checkpoint_directory}/samples', f'{n_samples}.png'))         
                    n_samples += 1


    
