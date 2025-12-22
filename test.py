import torch
import os
import numpy as np
import hydra
from omegaconf import DictConfig

from networks import GlobalGenerator
from pipeline import get_dataloaders
from utils import set_seed, load_checkpoint, save_comparison_images, compute_psnr, compute_ssim, save_npz


def test_step(dpd, mag, generator, device):
    """
    Single test step for one batch
    
    Args:
        dpd: Input tensor (B, 1, H, W)
        mag: Target tensor (B, 1, H, W)
        generator: Generator model
        device: torch device
    
    Returns:
        dict: Dictionary containing metrics and generated images
    """
    dpd = dpd.to(device)
    mag = mag.to(device)
    
    with torch.no_grad():
        # Generate
        generated = generator(dpd)
        
        # Compute metrics
        psnr = compute_psnr(generated, mag)
        ssim_val = compute_ssim(generated, mag)
    
    return {
        'psnr': psnr,
        'ssim': ssim_val,
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
    
    # 3. Create test results directory
    result_dir = f"{cfg.paths.result_root}/{cfg.experiment_name}"
    test_root = f"{result_dir}/test"
    os.makedirs(test_root, exist_ok=True)
    test_dir = f"{test_root}/epoch_{cfg.test.epoch:04d}"
    os.makedirs(test_dir, exist_ok=True)
    print(f"Test results will be saved to {test_dir}")
    
    # 4. Load test dataloader
    print("\nLoading test dataset...")
    _, _, test_loader = get_dataloaders(cfg)
    
    # 5. Initialize generator
    print("\nInitializing generator...")
    generator = GlobalGenerator(
        input_nc=cfg.model.generator.input_nc,
        output_nc=cfg.model.generator.output_nc,
        ngf=cfg.model.generator.ngf,
        n_downsampling=cfg.model.generator.n_downsampling,
        n_blocks=cfg.model.generator.n_blocks
    ).to(device)
    
    # 6. Load checkpoint
    checkpoint_dir = f"{result_dir}/checkpoints"
    checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch_{cfg.validation.epoch:04d}.pth"

    if not os.path.exists(checkpoint_path):
        print(f"Error: No checkpoints found {checkpoint_path}")
        return
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    epoch = load_checkpoint(checkpoint_path, generator, None, None, None)
    
    # 7. Run inference
    print(f"\nRunning inference on test set...")
    generator.eval()
    
    all_psnr = []
    all_ssim = []

    file_list = test_loader.dataset.file_list
    file_idx = 0
    
    for batch_idx, batch in enumerate(test_loader):
        # Perform test step
        results = test_step(
            batch['dpd'],
            batch['mag'],
            generator,
            device
        )
        
        all_psnr.append(results['psnr'])
        all_ssim.append(results['ssim'])
        
        # Save comparison image
        file_name = os.path.splitext(os.path.basename(file_list[file_idx]))[0]
        save_comparison_images(
            results['dpd'],
            results['mag']*100.,
            results['generated']*100.,
            f"{test_dir}/{file_name}.png",
            epoch,
            batch_idx
        )
        save_npz(
            results['dpd'],
            results['mag']*100.,
            results['generated']*100.,
            f"{test_dir}/{file_name}.npz"
        )
        file_idx += 1
    
        # Print progress
        print(f"Sample {batch_idx + 1}/{len(test_loader)}: "
              f"PSNR={results['psnr']:.2f} dB, SSIM={results['ssim']:.4f}")
    
    # 8. Print final statistics
    avg_psnr = sum(all_psnr) / len(all_psnr) if all_psnr else 0
    avg_ssim = sum(all_ssim) / len(all_ssim) if all_ssim else 0
    
    print(f"\n{'='*80}")
    print(f"Test Results Summary:")
    print(f"  Checkpoint epoch: {epoch}")
    print(f"  Total samples: {len(all_psnr)}")
    print(f"  Average PSNR: {avg_psnr:.2f} dB")
    print(f"  Average SSIM: {avg_ssim:.4f}")
    print(f"  Results saved to: {test_dir}")
    print(f"{'='*80}\n")
    
    # Save summary to text file
    summary_path = f"{test_dir}/test_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Test Results Summary\n")
        f.write(f"{'='*80}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Epoch: {epoch}\n")
        f.write(f"Total samples: {len(all_psnr)}\n")
        f.write(f"Average PSNR: {avg_psnr:.2f} dB\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Per-sample results:\n")
        for i, (psnr, ssim_val) in enumerate(zip(all_psnr, all_ssim)):
            f.write(f"Sample {i}: PSNR={psnr:.2f} dB, SSIM={ssim_val:.4f}\n")
    
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()

# import torch
# import os
# import numpy as np
# import hydra
# from omegaconf import DictConfig

# from networks import GlobalGenerator
# from pipeline import get_dataloaders
# from utils import set_seed, load_checkpoint, save_comparison_images, compute_psnr, compute_ssim


# @hydra.main(config_path="./configs", config_name="config", version_base=None)
# def main(cfg: DictConfig):
#     # 1. Set random seed for reproducibility
#     set_seed(cfg.seed)
#     print(f"Random seed set to {cfg.seed}")
    
#     # 2. Setup device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
    
#     # 3. Create test results directory
#     test_dir = f"{cfg.paths.result_dir}/test_results"
#     os.makedirs(test_dir, exist_ok=True)
#     print(f"Test results will be saved to {test_dir}")
    
#     # 4. Load test dataloader
#     print("\nLoading test dataset...")
#     _, _, test_loader = get_dataloaders(cfg)
    
#     # 5. Initialize generator
#     print("\nInitializing generator...")
#     generator = GlobalGenerator(
#         input_nc=cfg.model.generator.input_nc,
#         output_nc=cfg.model.generator.output_nc,
#         ngf=cfg.model.generator.ngf,
#         n_downsampling=cfg.model.generator.n_downsampling,
#         n_blocks=cfg.model.generator.n_blocks
#     ).to(device)
    
#     # 6. Load checkpoint
#     checkpoint_path = f"{cfg.paths.result_dir}/checkpoints/best_model.pth"
    
#     if not os.path.exists(checkpoint_path):
#         print(f"Error: Checkpoint not found at {checkpoint_path}")
#         print("Please make sure you have trained the model first.")
#         return
    
#     print(f"Loading checkpoint from {checkpoint_path}...")
#     epoch = load_checkpoint(checkpoint_path, generator, None, None, None)
    
#     # 7. Run inference
#     print(f"\nRunning inference on test set...")
#     generator.eval()
    
#     all_psnr = []
#     all_ssim = []
    
#     with torch.no_grad():
#         for batch_idx, batch in enumerate(test_loader):
#             dpd = batch['dpd'].to(device)
#             mag = batch['mag'].to(device)
            
#             # Generate
#             generated = generator(dpd)
            
#             # Compute metrics
#             psnr = compute_psnr(generated, mag)
#             ssim_val = compute_ssim(generated, mag)
            
#             all_psnr.append(psnr)
#             all_ssim.append(ssim_val)
            
#             # Save comparison image
#             save_path = f"{test_dir}/test_sample_{batch_idx}.png"
#             save_comparison_images(
#                 dpd[0],
#                 mag[0],
#                 generated[0],
#                 save_path,
#                 epoch,
#                 batch_idx
#             )
            
#             # Save as npz
#             dpd_np = dpd[0].cpu().numpy()
#             mag_np = mag[0].cpu().numpy()
#             generated_np = generated[0].cpu().numpy()
            
#             npz_path = f"{test_dir}/test_sample_{batch_idx}.npz"
#             np.savez(
#                 npz_path,
#                 dpd=dpd_np,
#                 mag=mag_np,
#                 generated=generated_np
#             )
            
#             # Print progress
#             print(f"Sample {batch_idx + 1}/{len(test_loader)}: "
#                   f"PSNR={psnr:.2f} dB, SSIM={ssim_val:.4f}")
    
#     # 8. Print final statistics
#     avg_psnr = sum(all_psnr) / len(all_psnr) if all_psnr else 0
#     avg_ssim = sum(all_ssim) / len(all_ssim) if all_ssim else 0
    
#     print(f"\n{'='*80}")
#     print(f"Test Results Summary:")
#     print(f"  Total samples: {len(all_psnr)}")
#     print(f"  Average PSNR: {avg_psnr:.2f} dB")
#     print(f"  Average SSIM: {avg_ssim:.4f}")
#     print(f"  Results saved to: {test_dir}")
#     print(f"{'='*80}\n")
    
#     # Save summary to text file
#     summary_path = f"{test_dir}/test_summary.txt"
#     with open(summary_path, 'w') as f:
#         f.write(f"Test Results Summary\n")
#         f.write(f"{'='*80}\n")
#         f.write(f"Checkpoint: {checkpoint_path}\n")
#         f.write(f"Epoch: {epoch}\n")
#         f.write(f"Total samples: {len(all_psnr)}\n")
#         f.write(f"Average PSNR: {avg_psnr:.2f} dB\n")
#         f.write(f"Average SSIM: {avg_ssim:.4f}\n")
#         f.write(f"{'='*80}\n\n")
#         f.write(f"Per-sample results:\n")
#         for i, (psnr, ssim_val) in enumerate(zip(all_psnr, all_ssim)):
#             f.write(f"Sample {i}: PSNR={psnr:.2f} dB, SSIM={ssim_val:.4f}\n")
    
#     print(f"Summary saved to {summary_path}")


# if __name__ == "__main__":
#     main()