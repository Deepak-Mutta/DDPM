import os
import torch
import torchvision
from PIL import Image
from unet import UNet
from diffusion import Diffusion

def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def sample_images(num_images=16, image_size=64, device="cuda"):
    model = UNet().to(device)
    diffusion = Diffusion(img_size=image_size, device="cuda")

    ckpt_path = os.path.join("models","DDPM_Uncondtional", "ckpt.pt")
    model.load_state_dict(torch.load(ckpt_path,map_location=device))
    model.eval()

    with torch.no_grad():
        sampled_images = diffusion.sample(model,n=num_images)
    
    os.makedirs("generated", exist_ok=True)
    l = len(os.listdir("generated"))
    save_images(sampled_images, f"generated/sample_{l}.jpg", nrow=4)
    print(f"Images saved to generated/sample_{l}.jpg")

for i in range(5):
    sample_images(num_images=16, image_size=64, device="cuda")