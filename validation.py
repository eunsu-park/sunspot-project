import torch
import torch.nn as nn
import os
import hydra
from omegaconf import DictConfig

from networks import GlobalGenerator, MultiscaleDiscriminator
from pipeline import get_dataloaders
from utils import set_seed, load_checkpoint, save_comparison_images, compute_psnr, compute_ssim, save_npz


def validation_step(dpd, mag, generator, discriminator, criterionFeat, device):
    """
    Single validation step for one batch
    
    Args:
        dpd: Input tensor (B, 1, H, W)
        mag: Target tensor (B, 1, H, W)
        generator: Generator model
        discriminator: Discriminator model
        criterionFeat: Feature matching loss function
        device: torch device
    
    Returns:
        dict: Dictionary containing loss, metrics, and generated images
    """
    dpd = dpd.to(device)
    mag = mag.to(device)
    
    with torch.no_grad():
        # Generate
        generated = generator(dpd)
        
        # Compute feature matching loss
        input_real = torch.cat([dpd, mag], dim=1)
        pred_real = discriminator(input_real)
        
        input_fake = torch.cat([dpd, generated], dim=1)
        pred_fake = discriminator(input_fake)
        
        # Feature matching loss
        loss_feat = 0
        for i in range(len(pred_real)):
            for j in range(len(pred_real[i][1])):
                loss_feat += criterionFeat(pred_fake[i][1][j], pred_real[i][1][j])
        
        # Compute metrics for each sample in batch
        psnr_scores = []
        ssim_scores = []
        for i in range(dpd.size(0)):
            psnr = compute_psnr(generated[i:i+1], mag[i:i+1])
            ssim_val = compute_ssim(generated[i:i+1], mag[i:i+1])
            psnr_scores.append(psnr)
            ssim_scores.append(ssim_val)
    
    return {
        'loss_feat': loss_feat.item(),
        'psnr': psnr_scores,
        'ssim': ssim_scores,
        'generated': generated,
        'dpd': dpd,
        'mag': mag
    }


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # 1. Set random seed for reproducibility
    set_seed(cfg.seed)
    print(f"Random seed set to {cfg.seed}")
    
    # 2. Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 3. Create validation results directory

    result_dir = f"{cfg.paths.result_root}/{cfg.experiment_name}"
    validation_root = f"{result_dir}/validation"
    os.makedirs(validation_root, exist_ok=True)
    validation_dir = f"{validation_root}/epoch_{cfg.validation.epoch:04d}"
    os.makedirs(validation_dir, exist_ok=True)
    print(f"Validation results will be saved to {validation_dir}")
    
    # 4. Load validation dataloader
    print("\nLoading validation dataset...")
    _, val_loader, _ = get_dataloaders(cfg)
    
    # 5. Initialize models
    print("\nInitializing models...")
    generator = GlobalGenerator(
        input_nc=cfg.model.generator.input_nc,
        output_nc=cfg.model.generator.output_nc,
        ngf=cfg.model.generator.ngf,
        n_downsampling=cfg.model.generator.n_downsampling,
        n_blocks=cfg.model.generator.n_blocks
    ).to(device)
    
    discriminator = MultiscaleDiscriminator(
        input_nc=cfg.model.discriminator.input_nc,
        ndf=cfg.model.discriminator.ndf,
        n_layers=cfg.model.discriminator.n_layers,
        num_D=cfg.model.discriminator.num_D
    ).to(device)
    
    # 6. Load checkpoint
    checkpoint_dir = f"{result_dir}/checkpoints"
    checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch_{cfg.validation.epoch:04d}.pth"

    if not os.path.exists(checkpoint_path):
        print(f"Error: No checkpoints found {checkpoint_path}")
        return
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    epoch = load_checkpoint(checkpoint_path, generator, discriminator, None, None)
    
    # 7. Setup loss function
    criterionFeat = nn.L1Loss()
    
    # 8. Run validation
    print(f"\nRunning validation...")
    generator.eval()
    discriminator.eval()
    
    all_losses = []
    all_psnr = []
    all_ssim = []

    file_list = val_loader.dataset.file_list
    file_idx = 0

    for batch_idx, batch in enumerate(val_loader):
        # Perform validation step
        results = validation_step(
            batch['dpd'],
            batch['mag'],
            generator,
            discriminator,
            criterionFeat,
            device
        )

        all_losses.append(results['loss_feat'])
        all_psnr.extend(results['psnr'])
        all_ssim.extend(results['ssim'])
        
        # Save all validation samples
        for i in range(results['dpd'].size(0)):
            file_name = os.path.splitext(os.path.basename(file_list[file_idx]))[0]
            # save_path = f"{validation_dir}/val_batch_{batch_idx}_sample_{i}.png"
            save_comparison_images(
                results['dpd'][i],
                results['mag'][i]*3000.,
                results['generated'][i]*3000.,
                f"{validation_dir}/{file_name}.png",
                epoch,
                f"{batch_idx}_{i}"
            )
            save_npz(
                results['dpd'][i],
                results['mag'][i]*3000.,
                results['generated'][i]*3000.,
                f"{validation_dir}/{file_name}.npz"
            )
            file_idx += 1
        
        # Print progress
        batch_psnr = sum(results['psnr']) / len(results['psnr'])
        batch_ssim = sum(results['ssim']) / len(results['ssim'])
        print(f"Batch [{batch_idx + 1}/{len(val_loader)}], "
              f"Loss: {results['loss_feat']:.4f}, "
              f"PSNR: {batch_psnr:.2f} dB, "
              f"SSIM: {batch_ssim:.4f}")
    
    # 9. Print final statistics
    avg_loss = sum(all_losses) / len(all_losses) if all_losses else 0
    avg_psnr = sum(all_psnr) / len(all_psnr) if all_psnr else 0
    avg_ssim = sum(all_ssim) / len(all_ssim) if all_ssim else 0
    
    print(f"\n{'='*80}")
    print(f"Validation Results Summary:")
    print(f"  Checkpoint epoch: {epoch}")
    print(f"  Total samples: {len(all_psnr)}")
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  Average PSNR: {avg_psnr:.2f} dB")
    print(f"  Average SSIM: {avg_ssim:.4f}")
    print(f"  Results saved to: {validation_dir}")
    print(f"{'='*80}\n")
    
    # Save summary to text file
    summary_path = f"{validation_dir}/validation_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Validation Results Summary\n")
        f.write(f"{'='*80}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Epoch: {epoch}\n")
        f.write(f"Total samples: {len(all_psnr)}\n")
        f.write(f"Average Loss: {avg_loss:.4f}\n")
        f.write(f"Average PSNR: {avg_psnr:.2f} dB\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Per-batch results:\n")
        batch_size = cfg.training.batch_size
        for i, loss in enumerate(all_losses):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(all_psnr))
            batch_psnr = sum(all_psnr[start_idx:end_idx]) / (end_idx - start_idx)
            batch_ssim = sum(all_ssim[start_idx:end_idx]) / (end_idx - start_idx)
            f.write(f"Batch {i}: Loss={loss:.4f}, PSNR={batch_psnr:.2f} dB, SSIM={batch_ssim:.4f}\n")
    
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()

# import torch
# import os
# from utils import save_comparison_images, compute_psnr, compute_ssim


# def validate_epoch(generator, val_loader, criterionFeat, discriminator, cfg, epoch, device, save_all=False):
#     """
#     Validate the model on validation set
    
#     Args:
#         generator: Generator model
#         val_loader: Validation dataloader
#         criterionFeat: Feature matching loss criterion
#         discriminator: Discriminator model (for feature extraction)
#         cfg: Config object
#         epoch: Current epoch number
#         device: torch device
#         save_all: If True, save all validation samples
    
#     Returns:
#         avg_loss: Average validation loss
#     """
#     generator.eval()
#     discriminator.eval()
    
#     val_losses = []
#     psnr_scores = []
#     ssim_scores = []
    
#     sample_dir = f"{cfg.paths.result_dir}/samples"
#     os.makedirs(sample_dir, exist_ok=True)
    
#     with torch.no_grad():
#         for batch_idx, batch in enumerate(val_loader):
#             dpd = batch['dpd'].to(device)
#             mag = batch['mag'].to(device)
            
#             # Generate
#             generated = generator(dpd)
            
#             # Compute feature matching loss
#             # Get features from real images
#             input_real = torch.cat([dpd, mag], dim=1)
#             pred_real = discriminator(input_real)
            
#             # Get features from fake images
#             input_fake = torch.cat([dpd, generated], dim=1)
#             pred_fake = discriminator(input_fake)
            
#             # Feature matching loss
#             loss_feat = 0
#             for i in range(len(pred_real)):
#                 for j in range(len(pred_real[i][1])):
#                     loss_feat += criterionFeat(pred_fake[i][1][j], pred_real[i][1][j].detach())
            
#             val_losses.append(loss_feat.item())
            
#             # Compute metrics for each sample in batch
#             for i in range(dpd.size(0)):
#                 psnr = compute_psnr(generated[i:i+1], mag[i:i+1])
#                 ssim_val = compute_ssim(generated[i:i+1], mag[i:i+1])
#                 psnr_scores.append(psnr)
#                 ssim_scores.append(ssim_val)
            
#             # Save comparison images
#             if save_all:
#                 # Save all samples in the batch
#                 for i in range(dpd.size(0)):
#                     save_path = f"{sample_dir}/val_epoch_{epoch}_batch_{batch_idx}_sample_{i}.png"
#                     save_comparison_images(
#                         dpd[i], 
#                         mag[i], 
#                         generated[i], 
#                         save_path, 
#                         epoch, 
#                         f"{batch_idx}_{i}"
#                     )
#             elif batch_idx == 0:
#                 # Save only first batch
#                 for i in range(min(4, dpd.size(0))):  # Save up to 4 samples
#                     save_path = f"{sample_dir}/val_epoch_{epoch}_sample_{i}.png"
#                     save_comparison_images(
#                         dpd[i], 
#                         mag[i], 
#                         generated[i], 
#                         save_path, 
#                         epoch, 
#                         i
#                     )
    
#     # Compute averages
#     avg_loss = sum(val_losses) / len(val_losses) if val_losses else 0
#     avg_psnr = sum(psnr_scores) / len(psnr_scores) if psnr_scores else 0
#     avg_ssim = sum(ssim_scores) / len(ssim_scores) if ssim_scores else 0
    
#     # Print results
#     print(f"Validation - Epoch [{epoch}/{cfg.training.num_epochs}], "
#           f"Avg Loss: {avg_loss:.4f}, PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}")
    
#     generator.train()
#     discriminator.train()
    
#     return avg_loss