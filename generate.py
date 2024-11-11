import torch 
import torchvision
import os
import argparse
import json
import numpy as np


from model import Generator
from utils import load_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Normalizing Flow.')
    parser.add_argument("--batch_size", type=int, default=2048,
                      help="The batch size to use for training.")
    args = parser.parse_args()


    with open("models/data.json", "r") as file:
        data = json.load(file)

    print('Model Loading...')
    # Model Pipeline
    mnist_dim = 784

    model = Generator(g_output_dim = mnist_dim).cuda()
    model = load_model(model, 'checkpoints')
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    print('Model loaded.')

    print('Start Generating')
    os.makedirs('samples', exist_ok=True)

    samples_per_class = [1200, 1200, 500, 1200, 1000, 500, 1200, 1000, 1200, 1000]

    num_class = 0
    with torch.no_grad():
        while num_class<10:
            num_samples = samples_per_class[num_class]
            mean_vector = np.array(data[str(num_class)][0])
            cov_matrix = np.array(data[str(num_class)][1])
            new_samples = np.random.multivariate_normal(mean_vector, cov_matrix, num_samples)
            new_samples_tensor = torch.tensor(new_samples, dtype=torch.float32).to('cuda')
            x = model(new_samples_tensor)
            x = x.reshape(num_samples, 28, 28)
            for k in range(x.shape[0]):
                if k<num_samples:
                    torchvision.utils.save_image(x[k:k+1], os.path.join('samples', f'{num_class * 1000 + k}.png'))      
                    #np.save(os.path.join('samples', f'{num_class * 1000 + k}_latent'), new_samples[k])   
            num_class += 1


    
