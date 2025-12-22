import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(epoch, generator, discriminator, opt_g, opt_d, save_path, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_g_state_dict': opt_g.state_dict(),
        'optimizer_d_state_dict': opt_d.state_dict(),
    }
    
    torch.save(checkpoint, save_path)
    
    if is_best:
        best_path = os.path.join(os.path.dirname(save_path), 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"Best model saved to {best_path}")


def load_checkpoint(checkpoint_path, generator, discriminator=None, opt_g=None, opt_d=None):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    generator.load_state_dict(checkpoint['generator_state_dict'])
    
    if discriminator is not None:
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    if opt_g is not None:
        opt_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
    
    if opt_d is not None:
        opt_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
    
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from epoch {epoch}")
    
    return epoch


def denormalize(tensor, mean, std):
    """Denormalize tensor for visualization"""
    return tensor * std + mean


def save_comparison_images(input_img, target_img, generated_img, save_path, epoch, batch_idx):
    """
    Save comparison images: input | target | generated
    
    Args:
        input_img: (C, H, W) tensor
        target_img: (C, H, W) tensor
        generated_img: (C, H, W) tensor
        save_path: path to save the image
        epoch: current epoch
        batch_idx: current batch index
    """
    # Convert to numpy and squeeze channel dimension
    input_np = input_img.detach().cpu().squeeze().numpy()
    target_np = target_img.detach().cpu().squeeze().numpy()
    generated_np = generated_img.detach().cpu().squeeze().numpy()
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot input
    im0 = axes[0].imshow(input_np, cmap='gray')
    axes[0].set_title('Input (DPD)')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Plot target
    im1 = axes[1].imshow(target_np, cmap='gray')
    axes[1].set_title('Target (MAG)')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Plot generated
    im2 = axes[2].imshow(generated_np, cmap='gray')
    axes[2].set_title('Generated (MAG)')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.suptitle(f'Epoch {epoch}, Batch {batch_idx}', fontsize=14)
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def save_npz(input_img, target_img, generated_img, save_path):
    """
    Save npz file: input | target | generated
    
    Args:
        input_img: (C, H, W) tensor
        target_img: (C, H, W) tensor
        generated_img: (C, H, W) tensor
        save_path: path to save the npz file
    """
    # Convert to numpy and squeeze channel dimension
    input_np = input_img.detach().cpu().squeeze().numpy()
    target_np = target_img.detach().cpu().squeeze().numpy()
    generated_np = generated_img.detach().cpu().squeeze().numpy()

    np.savez(save_path, inp=input_np, tar=target_np,
             gen=generated_np)


def compute_psnr(img1, img2):
    """
    Compute PSNR between two images
    
    Args:
        img1: (B, C, H, W) or (C, H, W) tensor
        img2: (B, C, H, W) or (C, H, W) tensor
    
    Returns:
        Average PSNR value
    """
    img1_np = img1.detach().cpu().numpy()
    img2_np = img2.detach().cpu().numpy()
    
    # Handle batch dimension
    if img1_np.ndim == 4:
        psnr_values = []
        for i in range(img1_np.shape[0]):
            # Remove channel dimension and compute
            val = psnr(img1_np[i, 0], img2_np[i, 0], data_range=img2_np[i, 0].max() - img2_np[i, 0].min())
            psnr_values.append(val)
        return np.mean(psnr_values)
    else:
        # Single image
        return psnr(img1_np[0], img2_np[0], data_range=img2_np[0].max() - img2_np[0].min())


def compute_ssim(img1, img2):
    """
    Compute SSIM between two images
    
    Args:
        img1: (B, C, H, W) or (C, H, W) tensor
        img2: (B, C, H, W) or (C, H, W) tensor
    
    Returns:
        Average SSIM value
    """
    img1_np = img1.detach().cpu().numpy()
    img2_np = img2.detach().cpu().numpy()
    
    # Handle batch dimension
    if img1_np.ndim == 4:
        ssim_values = []
        for i in range(img1_np.shape[0]):
            # Remove channel dimension and compute
            val = ssim(img1_np[i, 0], img2_np[i, 0], data_range=img2_np[i, 0].max() - img2_np[i, 0].min())
            ssim_values.append(val)
        return np.mean(ssim_values)
    else:
        # Single image
        return ssim(img1_np[0], img2_np[0], data_range=img2_np[0].max() - img2_np[0].min())