from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import os

# Hyperparameters
batchSize = 64  # Batch size
imageSize = 64  # Generated image size (64x64)

# Data transformations
transform = transforms.Compose([
    transforms.Resize(imageSize),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Weight initialization function
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Generator class
class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# Discriminator class
class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1)

if __name__ == '__main__':
    # Load dataset and dataloader
    dataset = dset.CIFAR10(root='./data', download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=True, num_workers=2)

    # Initialize the generator and discriminator
    netG = G()
    netG.apply(weights_init)
    netD = D()
    netD.apply(weights_init)

    # Loss function and optimizers
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Directory for results
    os.makedirs("results", exist_ok=True)

    # Training loop
    for epoch in range(25):
        for i, data in enumerate(dataloader, 0):
            # 1st Step: Update Discriminator
            netD.zero_grad()
            real, _ = data
            target_real = torch.ones(real.size(0))
            output_real = netD(real)
            errD_real = criterion(output_real, target_real)

            # Train discriminator on fake data
            noise = torch.randn(real.size(0), 100, 1, 1)
            fake = netG(noise)
            target_fake = torch.zeros(real.size(0))
            output_fake = netD(fake.detach())
            errD_fake = criterion(output_fake, target_fake)

            # Backpropagation for discriminator
            errD = errD_real + errD_fake
            errD.backward()
            optimizerD.step()

            # 2nd Step: Update Generator
            netG.zero_grad()
            target = torch.ones(real.size(0))
            output = netD(fake)
            errG = criterion(output, target)
            errG.backward()
            optimizerG.step()

            # Print losses and save images periodically
            print(f'[{epoch+1}/{25}][{i+1}/{len(dataloader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}')
            if i % 100 == 0:
                vutils.save_image(real, 'results/real_samples.png', normalize=True)
                fake = netG(noise)
                vutils.save_image(fake.data, f'results/fake_samples_epoch_{epoch+1:03d}.png', normalize=True)
