import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class NPZDataset(Dataset):
    """Dataset for loading NPZ files containing 'dpd' and 'mag' arrays"""
    
    def __init__(self, data_dir, normalization_config):
        """
        Args:
            data_dir: Directory containing NPZ files
            normalization_config: Config dict with dpd/mag mean and std
        """
        self.data_dir = data_dir
        self.norm_config = normalization_config
        
        # Scan for all npz files
        self.file_list = sorted(list(Path(data_dir).glob('*.npz')))
        
        if len(self.file_list) == 0:
            raise ValueError(f"No NPZ files found in {data_dir}")
        
        print(f"Found {len(self.file_list)} NPZ files in {data_dir}")
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        """
        Returns:
            dict: {'dpd': tensor (1, 512, 512), 'mag': tensor (1, 512, 512)}
        """
        # Load npz file
        npz_path = self.file_list[idx]
        data = np.load(npz_path)
        
        dpd = data['dpd']  # (512, 512)
        mag = data['mag']  # (512, 512)
        
        # Normalize
        # dpd_norm = self._normalize(dpd, 
        #                            self.norm_config.dpd_mean, 
        #                            self.norm_config.dpd_std)
        dpd_norm = dpd
        # mag_norm = self._normalize(mag, 
        #                            self.norm_config.mag_mean, 
        #                            self.norm_config.mag_std)
        mag_norm = np.clip(mag/3000., -1, 1)

        
        # Convert to tensor and add channel dimension
        dpd_tensor = torch.from_numpy(dpd_norm).float().unsqueeze(0)  # (1, 512, 512)
        mag_tensor = torch.from_numpy(mag_norm).float().unsqueeze(0)  # (1, 512, 512)
        
        return {
            'dpd': dpd_tensor,
            'mag': mag_tensor
        }
    
    def _normalize(self, data, mean, std):
        """
        Normalize data using mean and std
        You can modify this function to use different normalization strategies
        
        Args:
            data: numpy array
            mean: mean value for normalization
            std: std value for normalization
        
        Returns:
            normalized numpy array
        """
        return (data - mean) / std


def get_dataloaders(cfg):
    """
    Create train, validation, and test dataloaders
    
    Args:
        cfg: Hydra config object
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = NPZDataset(
        data_dir=cfg.paths.data.train_dir,
        normalization_config=cfg.normalization
    )
    
    val_dataset = NPZDataset(
        data_dir=cfg.paths.data.val_dir,
        normalization_config=cfg.normalization
    )
    
    test_dataset = NPZDataset(
        data_dir=cfg.paths.data.test_dir,
        normalization_config=cfg.normalization
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=cfg.dataloader.shuffle,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one at a time for testing
        shuffle=False,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader