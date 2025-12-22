import torch
import torch.nn as nn
import torch.optim as optim
import os, time
import hydra
from omegaconf import DictConfig

from networks import GlobalGenerator, MultiscaleDiscriminator, GANLoss, get_pix2pixhd_scheduler
from pipeline import get_dataloaders
from utils import set_seed, save_checkpoint, save_comparison_images


def train_step(dpd, mag, generator, discriminator, opt_g, opt_d, criterionGAN, criterionFeat, lambda_feat, device):
    """
    Single training step for one batch
    
    Args:
        dpd: Input tensor (B, 1, H, W)
        mag: Target tensor (B, 1, H, W)
        generator: Generator model
        discriminator: Discriminator model
        opt_g: Generator optimizer
        opt_d: Discriminator optimizer
        criterionGAN: GAN loss function
        criterionFeat: Feature matching loss function
        lambda_feat: Weight for feature matching loss
        device: torch device
    
    Returns:
        dict: Dictionary containing loss values
    """
    dpd = dpd.to(device)
    mag = mag.to(device)
    
    # ============================================
    # (1) Update Discriminator
    # ============================================
    opt_d.zero_grad()
    
    # Real images
    input_real = torch.cat([dpd, mag], dim=1)
    pred_real = discriminator(input_real)
    loss_d_real = 0
    for pred in pred_real:
        loss_d_real += criterionGAN(pred[0], True)
    
    # Fake images
    generated = generator(dpd)
    input_fake = torch.cat([dpd, generated.detach()], dim=1)
    pred_fake = discriminator(input_fake)
    loss_d_fake = 0
    for pred in pred_fake:
        loss_d_fake += criterionGAN(pred[0], False)
    
    # Total discriminator loss
    loss_d = (loss_d_real + loss_d_fake) * 0.5
    loss_d.backward()
    opt_d.step()
    
    # ============================================
    # (2) Update Generator
    # ============================================
    opt_g.zero_grad()
    
    # GAN loss
    input_fake = torch.cat([dpd, generated], dim=1)
    pred_fake = discriminator(input_fake)
    loss_g_gan = 0
    for pred in pred_fake:
        loss_g_gan += criterionGAN(pred[0], True)
    
    # Feature matching loss
    loss_g_feat = 0
    for i in range(len(pred_real)):
        for j in range(len(pred_real[i][1])):
            loss_g_feat += criterionFeat(pred_fake[i][1][j], pred_real[i][1][j].detach())
    
    # Total generator loss
    loss_g = loss_g_gan + lambda_feat * loss_g_feat
    loss_g.backward()
    opt_g.step()
    
    return {
        'loss_d': loss_d.item(),
        'loss_g': loss_g.item(),
        'loss_g_gan': loss_g_gan.item(),
        'loss_g_feat': loss_g_feat.item(),
        'generated': generated.detach()
    }


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # 1. Set random seed for reproducibility
    set_seed(cfg.seed)
    print(f"Random seed set to {cfg.seed}")

    result_dir = f"{cfg.paths.result_root}/{cfg.experiment_name}"
    
    # 2. Create directories
    checkpoint_dir = f"{result_dir}/checkpoints"
    sample_dir = f"{result_dir}/samples"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    print(f"Results will be saved to {result_dir}")
    
    # 3. Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 4. Create dataloaders
    print("\nLoading datasets...")
    train_loader, _, _ = get_dataloaders(cfg)
    
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
    
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # 6. Setup optimizers
    opt_g = optim.Adam(
        generator.parameters(),
        lr=cfg.training.lr_g,
        betas=(cfg.training.beta1, cfg.training.beta2)
    )
    
    opt_d = optim.Adam(
        discriminator.parameters(),
        lr=cfg.training.lr_d,
        betas=(cfg.training.beta1, cfg.training.beta2)
    )

    scheduler_g = get_pix2pixhd_scheduler(opt_g, cfg.training.num_epochs, cfg.training.num_epochs//2)
    scheduler_d = get_pix2pixhd_scheduler(opt_d, cfg.training.num_epochs, cfg.training.num_epochs//2)
    
    # 7. Setup loss functions
    criterionGAN = GANLoss().to(device)
    criterionFeat = nn.L1Loss()
    
    # 8. Training loop
    print("\nStarting training...")
    
    for epoch in range(1, cfg.training.num_epochs + 1):

        generator.train()
        discriminator.train()

        t0 = time.time()
        epoch_d_loss = 0
        epoch_g_loss = 0
        epoch_g_gan = 0
        epoch_g_feat = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Perform training step
            losses = train_step(
                batch['dpd'],
                batch['mag'],
                generator,
                discriminator,
                opt_g,
                opt_d,
                criterionGAN,
                criterionFeat,
                cfg.loss.lambda_feat,
                device
            )

            # Accumulate losses
            epoch_d_loss += losses['loss_d']
            epoch_g_loss += losses['loss_g']
            epoch_g_gan += losses['loss_g_gan']
            epoch_g_feat += losses['loss_g_feat']

            if batch_idx % cfg.checkpoint.report_freq == 0: 
                print(f"Epoch [{epoch}/{cfg.training.num_epochs}], "
                      f"Batch [{batch_idx + 1}/{len(train_loader)}], "
                      f"D_loss: {losses['loss_d']:.4f}, "
                      f"G_loss: {losses['loss_g']:.4f}, "
                      f"G_GAN: {losses['loss_g_gan']:.4f}, "
                      f"G_Feat: {losses['loss_g_feat']:.4f}, "
                      f"Elapsed Time: {int(time.time() - t0):d}")
                
        # Epoch summary
        avg_d_loss = epoch_d_loss / len(train_loader)
        avg_g_loss = epoch_g_loss / len(train_loader)
        avg_g_gan = epoch_g_gan / len(train_loader)
        avg_g_feat = epoch_g_feat / len(train_loader)
        
        print(f"\n{'='*80}")
        print(f"Epoch [{epoch}/{cfg.training.num_epochs}] Summary:")
        print(f"  Avg D_loss: {avg_d_loss:.4f}")
        print(f"  Avg G_loss: {avg_g_loss:.4f} (GAN: {avg_g_gan:.4f}, Feat: {avg_g_feat:.4f})")
        print(f"Elapsed time: {int(time.time() - t0):d} sec")
        print(f"{'='*80}\n")
        
        # Save sample images periodically
        if epoch % cfg.checkpoint.report_freq == 0:
            print(f"Saving sample images for epoch {epoch}...")
            generator.eval()
            with torch.no_grad():
                # Get first batch for visualization
                sample_batch = next(iter(train_loader))
                sample_dpd = sample_batch['dpd'].to(device)
                sample_mag = sample_batch['mag'].to(device)
                sample_generated = generator(sample_dpd)
                
                # Save up to 4 samples
                for i in range(min(4, sample_dpd.size(0))):
                    save_path = f"{sample_dir}/train_epoch_{epoch:04d}_sample_{i}.png"
                    save_comparison_images(
                        sample_dpd[i],
                        sample_mag[i],
                        sample_generated[i],
                        save_path,
                        epoch,
                        i
                    )
            generator.train()
        
        # Save checkpoint periodically
        if epoch % cfg.checkpoint.save_freq == 0:
            save_path = f"{checkpoint_dir}/checkpoint_epoch_{epoch:04d}.pth"
            save_checkpoint(
                epoch, 
                generator, 
                discriminator, 
                opt_g, 
                opt_d, 
                save_path,
                is_best=False
            )
            print(f"Checkpoint saved at epoch {epoch}")

        scheduler_g.step()
        scheduler_d.step()
        
        print()  # Empty line for readability
    
    print("Training completed!")
    print(f"Results saved to {result_dir}")


if __name__ == "__main__":
    main()

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import os
# import hydra
# from omegaconf import DictConfig

# from networks import GlobalGenerator, MultiscaleDiscriminator, GANLoss
# from pipeline import get_dataloaders
# from validation import validate_epoch
# from utils import set_seed, save_checkpoint, save_comparison_images


# @hydra.main(config_path="./configs", config_name="config", version_base=None)
# def main(cfg: DictConfig):
#     # 1. Set random seed for reproducibility
#     set_seed(cfg.seed)
#     print(f"Random seed set to {cfg.seed}")
    
#     # 2. Create directories
#     checkpoint_dir = f"{cfg.paths.result_dir}/checkpoints"
#     sample_dir = f"{cfg.paths.result_dir}/samples"
#     os.makedirs(checkpoint_dir, exist_ok=True)
#     os.makedirs(sample_dir, exist_ok=True)
#     print(f"Results will be saved to {cfg.paths.result_dir}")
    
#     # 3. Setup device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
    
#     # 4. Create dataloaders
#     print("\nLoading datasets...")
#     train_loader, val_loader, _ = get_dataloaders(cfg)
    
#     # 5. Initialize models
#     print("\nInitializing models...")
#     generator = GlobalGenerator(
#         input_nc=cfg.model.generator.input_nc,
#         output_nc=cfg.model.generator.output_nc,
#         ngf=cfg.model.generator.ngf,
#         n_downsampling=cfg.model.generator.n_downsampling,
#         n_blocks=cfg.model.generator.n_blocks
#     ).to(device)
    
#     discriminator = MultiscaleDiscriminator(
#         input_nc=cfg.model.discriminator.input_nc,
#         ndf=cfg.model.discriminator.ndf,
#         n_layers=cfg.model.discriminator.n_layers,
#         num_D=cfg.model.discriminator.num_D
#     ).to(device)
    
#     print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
#     print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
#     # 6. Setup optimizers
#     opt_g = optim.Adam(
#         generator.parameters(),
#         lr=cfg.training.lr_g,
#         betas=(cfg.training.beta1, cfg.training.beta2)
#     )
    
#     opt_d = optim.Adam(
#         discriminator.parameters(),
#         lr=cfg.training.lr_d,
#         betas=(cfg.training.beta1, cfg.training.beta2)
#     )
    
#     # 7. Setup loss functions
#     criterionGAN = GANLoss().to(device)
#     criterionFeat = nn.L1Loss()
    
#     # 8. Training loop
#     print("\nStarting training...")
#     best_val_loss = float('inf')
    
#     for epoch in range(1, cfg.training.num_epochs + 1):
#         generator.train()
#         discriminator.train()
        
#         epoch_d_loss = 0
#         epoch_g_loss = 0
#         epoch_g_gan = 0
#         epoch_g_feat = 0
        
#         for batch_idx, batch in enumerate(train_loader):
#             dpd = batch['dpd'].to(device)
#             mag = batch['mag'].to(device)
            
#             batch_size = dpd.size(0)
            
#             # ============================================
#             # (1) Update Discriminator
#             # ============================================
#             opt_d.zero_grad()
            
#             # Real images
#             input_real = torch.cat([dpd, mag], dim=1)
#             pred_real = discriminator(input_real)
#             loss_d_real = 0
#             for pred in pred_real:
#                 loss_d_real += criterionGAN(pred[0], True)
            
#             # Fake images
#             generated = generator(dpd)
#             input_fake = torch.cat([dpd, generated.detach()], dim=1)
#             pred_fake = discriminator(input_fake)
#             loss_d_fake = 0
#             for pred in pred_fake:
#                 loss_d_fake += criterionGAN(pred[0], False)
            
#             # Total discriminator loss
#             loss_d = (loss_d_real + loss_d_fake) * 0.5
#             loss_d.backward()
#             opt_d.step()
            
#             # ============================================
#             # (2) Update Generator
#             # ============================================
#             opt_g.zero_grad()
            
#             # GAN loss
#             input_fake = torch.cat([dpd, generated], dim=1)
#             pred_fake = discriminator(input_fake)
#             loss_g_gan = 0
#             for pred in pred_fake:
#                 loss_g_gan += criterionGAN(pred[0], True)
            
#             # Feature matching loss
#             loss_g_feat = 0
#             for i in range(len(pred_real)):
#                 for j in range(len(pred_real[i][1])):
#                     loss_g_feat += criterionFeat(pred_fake[i][1][j], pred_real[i][1][j].detach())
            
#             # Total generator loss
#             loss_g = loss_g_gan + cfg.loss.lambda_feat * loss_g_feat
#             loss_g.backward()
#             opt_g.step()
            
#             # Accumulate losses
#             epoch_d_loss += loss_d.item()
#             epoch_g_loss += loss_g.item()
#             epoch_g_gan += loss_g_gan.item()
#             epoch_g_feat += loss_g_feat.item()
            
#             # Print progress
#             if (batch_idx + 1) % 10 == 0:
#                 print(f"Epoch [{epoch}/{cfg.training.num_epochs}], "
#                       f"Batch [{batch_idx + 1}/{len(train_loader)}], "
#                       f"D_loss: {loss_d.item():.4f}, "
#                       f"G_loss: {loss_g.item():.4f}, "
#                       f"G_GAN: {loss_g_gan.item():.4f}, "
#                       f"G_Feat: {loss_g_feat.item():.4f}")
        
#         # Epoch summary
#         avg_d_loss = epoch_d_loss / len(train_loader)
#         avg_g_loss = epoch_g_loss / len(train_loader)
#         avg_g_gan = epoch_g_gan / len(train_loader)
#         avg_g_feat = epoch_g_feat / len(train_loader)
        
#         print(f"\n{'='*80}")
#         print(f"Epoch [{epoch}/{cfg.training.num_epochs}] Summary:")
#         print(f"  Avg D_loss: {avg_d_loss:.4f}")
#         print(f"  Avg G_loss: {avg_g_loss:.4f} (GAN: {avg_g_gan:.4f}, Feat: {avg_g_feat:.4f})")
#         print(f"{'='*80}\n")
        
#         # Save sample images periodically
#         if epoch % cfg.checkpoint.sample_freq == 0:
#             print(f"Saving sample images for epoch {epoch}...")
#             generator.eval()
#             with torch.no_grad():
#                 # Get first batch for visualization
#                 sample_batch = next(iter(train_loader))
#                 sample_dpd = sample_batch['dpd'].to(device)
#                 sample_mag = sample_batch['mag'].to(device)
#                 sample_generated = generator(sample_dpd)
                
#                 # Save up to 4 samples
#                 for i in range(min(4, sample_dpd.size(0))):
#                     save_path = f"{sample_dir}/train_epoch_{epoch}_sample_{i}.png"
#                     save_comparison_images(
#                         sample_dpd[i],
#                         sample_mag[i],
#                         sample_generated[i],
#                         save_path,
#                         epoch,
#                         i
#                     )
#             generator.train()
        
#         # Validation
#         if epoch % cfg.checkpoint.sample_freq == 0:
#             print(f"\nRunning validation for epoch {epoch}...")
#             val_loss = validate_epoch(
#                 generator, 
#                 val_loader, 
#                 criterionFeat,
#                 discriminator,
#                 cfg, 
#                 epoch, 
#                 device,
#                 save_all=False
#             )
            
#             # Save best model
#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 save_path = f"{checkpoint_dir}/checkpoint_epoch_{epoch}.pth"
#                 save_checkpoint(
#                     epoch, 
#                     generator, 
#                     discriminator, 
#                     opt_g, 
#                     opt_d, 
#                     save_path,
#                     is_best=True
#                 )
#                 print(f"New best model saved with validation loss: {val_loss:.4f}")
        
#         # Save checkpoint periodically
#         if epoch % cfg.checkpoint.save_freq == 0:
#             save_path = f"{checkpoint_dir}/checkpoint_epoch_{epoch}.pth"
#             save_checkpoint(
#                 epoch, 
#                 generator, 
#                 discriminator, 
#                 opt_g, 
#                 opt_d, 
#                 save_path,
#                 is_best=False
#             )
#             print(f"Checkpoint saved at epoch {epoch}")
        
#         print()  # Empty line for readability
    
#     print("Training completed!")
#     print(f"Best validation loss: {best_val_loss:.4f}")
#     print(f"Results saved to {cfg.paths.result_dir}")


# if __name__ == "__main__":
#     main()