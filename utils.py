import torch
from torch_fidelity import calculate_metrics
from torchvision import datasets, transforms
import torchvision.utils as vutils

import argparse
import os

def save_models(G, D, folder):
    torch.save(G.state_dict(), os.path.join(folder,'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder,'D.pth'))


def load_model(G, folder):
    ckpt = torch.load(os.path.join(folder,'G.pth'))
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G

def gradient_penalty(critic, real, fake, device='cpu'):
    batch_size, n_channels, h, w = real.shape

    epsilon = torch.rand((batch_size, 1, 1, 1)).repeat(1, n_channels, h, w).to(device)

    interpolated_images = real*epsilon + fake*(1-epsilon)

    mixed_scores = critic(interpolated_images)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gp = torch.mean((gradient_norm - 1) ** 2)
    
    return gp


def from_mnist_to_jpeg(real_images_path="data/train_images/"):
    os.makedirs(real_images_path, exist_ok=True)

    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5), std=(0.5))])
    train_dataset = datasets.MNIST(root='data/MNIST/', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=64, shuffle=True)

    for i, (images, _) in enumerate(train_loader):
        for j in range(images.size(0)):
            image_path = os.path.join(real_images_path, f"{i * len(images) + j}.png")
            vutils.save_image(images[j], image_path)

def metrics(real_images_path = "data/train_images", generated_images_path = "samples/"):
    metrics = calculate_metrics(
        input1=real_images_path,
        input2=generated_images_path,
        fid=True,
        precision=True,
        recall=True
    )
    print("FID:", metrics['frechet_inception_distance'])
    return metrics['frechet_inception_distance']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute Metrics.')
    parser.add_argument("--checkpoint_directory", type=str, default='checkpoints',
                      help="The directory where the model is located.")
    args = parser.parse_args()

    # from_mnist_to_jpeg()
    metrics(generated_images_path=f'{args.checkpoint_directory}/samples/')

# def D_train(x, G, D, D_optimizer, criterion):
#     #=======================Train the discriminator=======================#
#     D.zero_grad()

#     # train discriminator on real
#     x_real, y_real = x, torch.ones(x.shape[0], 1)
#     x_real, y_real = x_real.cuda(), y_real.cuda()

#     D_output = D(x_real)
#     D_real_loss = criterion(D_output, y_real)
#     D_real_score = D_output

#     # train discriminator on facke
#     z = torch.randn(x.shape[0], 100).cuda()
#     x_fake, y_fake = G(z), torch.zeros(x.shape[0], 1).cuda()

#     D_output =  D(x_fake)
    
#     D_fake_loss = criterion(D_output, y_fake)
#     D_fake_score = D_output

#     # gradient backprop & optimize ONLY D's parameters
#     D_loss = D_real_loss + D_fake_loss
#     D_loss.backward()
#     D_optimizer.step()
        
#     return  D_loss.data.item()


# def G_train(x, G, D, G_optimizer, criterion):
#     #=======================Train the generator=======================#
#     G.zero_grad()

#     z = torch.randn(x.shape[0], 100).cuda()
#     y = torch.ones(x.shape[0], 1).cuda()
                 
#     G_output = G(z)
#     D_output = D(G_output)
#     G_loss = criterion(D_output, y)

#     # gradient backprop & optimize ONLY G's parameters
#     G_loss.backward()
#     G_optimizer.step()
        
#     return G_loss.data.item()



# def save_models(G, D, folder):
#     torch.save(G.state_dict(), os.path.join(folder,'G.pth'))
#     torch.save(D.state_dict(), os.path.join(folder,'D.pth'))


# def load_model(G, folder):
#     ckpt = torch.load(os.path.join(folder,'G.pth'))
#     G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
#     return G
