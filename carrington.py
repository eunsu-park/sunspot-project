import torch
import os
from glob import glob
import numpy as np
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt

from networks import GlobalGenerator
from pipeline import get_dataloaders
from utils import set_seed, load_checkpoint, save_comparison_images, compute_psnr, compute_ssim, save_npz


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
    carrington_root = f"{result_dir}/carrington"
    os.makedirs(carrington_root, exist_ok=True)
    carrington_dir = f"{carrington_root}/epoch_{cfg.test.epoch:04d}"
    os.makedirs(carrington_dir, exist_ok=True)
    print(f"Test results will be saved to {carrington_dir}")

    file_list = sorted(glob(f"{cfg.paths.data.carrington_dir}/*.npy"))
    print(len(file_list))

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
    checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch_{cfg.carrington.epoch:04d}.pth"

    if not os.path.exists(checkpoint_path):
        print(f"Error: No checkpoints found {checkpoint_path}")
        return
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    epoch = load_checkpoint(checkpoint_path, generator, None, None, None)
    
    # 7. Run inference
    print(f"\nRunning inference on carrington set...")
    generator.eval()

    idx = 0

    for file_path in file_list :
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        # save_path = f"{carrington_dir}/{file_name}"
        data = np.load(file_path).astype(np.float64)
        data = np.expand_dims(data, 0)
        data = np.expand_dims(data, 0)
        data = data * (2./255.) - 1.
        data = torch.from_numpy(data).float().to(device)
        gen = generator(data)
        inp = data
        tar = torch.zeros_like(inp)
        gen = gen * 3000.

        save_comparison_images(
            inp,
            tar,
            gen,
            f"{carrington_dir}/{file_name}.png",
            epoch,
            idx
        )
        save_npz(
            inp,
            tar,
            gen,
            f"{carrington_dir}/{file_name}.npz"
        )        

        print(data.shape, data.min(), data.max(), file_name)
        print(gen.shape, gen.min(), gen.max())

        idx += 1


if __name__ == "__main__" :
    main()