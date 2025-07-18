import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
from PIL import Image
import logging
from torch import optim
from torch.utils.data import DataLoader
from unet import UNet
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device="cuda"):
        self.noise_steps=noise_steps
        self.beta_start=beta_start
        self.beta_end=beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.noise_scheduler().to(device)
        self.alpha = 1-self.beta
        self.alpha_hat = torch.cumprod(self.alpha,dim=0)

    def noise_scheduler(self):
        return torch.linspace(self.beta_start,self.beta_end,self.noise_steps)
    
    def forward_pass(self, x, t):
        root_alpha = torch.sqrt(self.alpha_hat[t])[:,None,None,None]
        root_1_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:,None,None,None]
        eps = torch.randn_like(x)
        return root_alpha*x + root_1_minus_alpha_hat*eps, eps
    
    def sample_time_steps(self,n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    
    def sample(self,model,n):
        logging.info(f"sampling {n} new images...")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n,3,self.img_size,self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1,self.noise_steps)),position =0 ):
                t = (torch.ones(n)*i).long().to(self.device)
                predicted_noise = model(x,t)
                alpha = self.alpha[t][:,None,None,None]
                alpha_hat = self.alpha_hat[t][:,None,None,None]
                beta = self.beta[t][:,None,None,None]

                if i>1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

def train(args):
    torch.cuda.empty_cache()
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_time_steps(images.shape[0]).to(device)
            x_t, noise = diffusion.forward_pass(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        sampled_images = diffusion.sample(model, n=images.shape[0])
        save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Uncondtional"
    args.epochs = 300
    args.batch_size = 4
    args.image_size = 64
    args.dataset_path = r"dataset"
    args.device = "cuda"
    args.lr = 3e-4
    train(args)

if __name__=="__main__":
    launch()