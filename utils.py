import torch
import os
from torch_fidelity import calculate_metrics
from torchvision import datasets, transforms
import torchvision.utils as vutils


def compute_gradient_penalty(D, real_samples, fake_samples):
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(real_samples.size(0), 1).cuda()
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(real_samples.size(0), 1).cuda()
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def D_train(x, G, D, D_optimizer, criterion, LAMBDA_GP=10):
    #=======================Train the critic=======================#
    D.zero_grad()

    # train discriminator on real
    x_real = x.cuda()
    D_real = D(x_real)
    
    # train discriminator on fake
    z = torch.randn(x.shape[0], 100).cuda()
    x_fake = G(z)
    D_fake = D(x_fake.detach())
    
    # Gradient penalty
    gradient_penalty = compute_gradient_penalty(D, x_real, x_fake.detach())
    
    # Wasserstein loss
    D_loss = -torch.mean(D_real) + torch.mean(D_fake) + LAMBDA_GP * gradient_penalty
    
    D_loss.backward()
    D_optimizer.step()
        
    return D_loss.data.item()


def G_train(x, G, D, G_optimizer, criterion):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(x.shape[0], 100).cuda()
    fake_imgs = G(z)
    G_loss = -torch.mean(D(fake_imgs))  # Wasserstein loss

    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.data.item()


def save_models(G, D, folder):
    torch.save(G.state_dict(), os.path.join(folder,'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder,'D.pth'))


def load_model(G, folder):
    ckpt = torch.load(os.path.join(folder,'G.pth'))
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G


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
    return

if __name__ == "__main__":
    # from_mnist_to_jpeg()
    metrics()
