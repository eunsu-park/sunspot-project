import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR


class GlobalGenerator(nn.Module):
    """
    Global Generator for pix2pix HD
    Architecture: Encoder -> Residual Blocks -> Decoder
    """
    
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=4, n_blocks=9):
        """
        Args:
            input_nc: number of input channels
            output_nc: number of output channels
            ngf: base number of filters
            n_downsampling: number of downsampling layers
            n_blocks: number of residual blocks
        """
        super(GlobalGenerator, self).__init__()
        
        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        ]
        
        # Downsampling (Encoder)
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(True)
            ]
        
        # Residual blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResidualBlock(ngf * mult)]
        
        # Upsampling (Decoder)
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, 
                                  kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(ngf * mult // 2),
                nn.ReLU(True)
            ]
        
        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)


class ResidualBlock(nn.Module):
    """Residual block with two conv layers"""
    
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels)
        )
    
    def forward(self, x):
        return x + self.block(x)


class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator for a single scale"""
    
    def __init__(self, input_nc, ndf=64, n_layers=3):
        """
        Args:
            input_nc: number of input channels (input + target concatenated)
            ndf: base number of filters
            n_layers: number of conv layers
        """
        super(NLayerDiscriminator, self).__init__()
        
        # First layer (no normalization)
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=2),
            nn.LeakyReLU(0.2, True)
        ]
        
        nf_mult = 1
        nf_mult_prev = 1
        
        # Intermediate layers
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=2),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        
        # Penultimate layer
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=2),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        # Final layer (output 1 channel for real/fake)
        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=2)
        ]
        
        self.model = nn.Sequential(*sequence)
    
    def forward(self, x):
        """
        Returns:
            list of intermediate feature maps and final output
        """
        results = [x]
        for submodel in self.model:
            intermediate = submodel(results[-1])
            results.append(intermediate)
        
        # Return final prediction and intermediate features
        return results[-1], results[1:-1]


class MultiscaleDiscriminator(nn.Module):
    """Multi-scale discriminator with num_D discriminators at different scales"""
    
    def __init__(self, input_nc, ndf=64, n_layers=3, num_D=3):
        """
        Args:
            input_nc: number of input channels
            ndf: base number of filters
            n_layers: number of layers in each discriminator
            num_D: number of discriminators at different scales
        """
        super(MultiscaleDiscriminator, self).__init__()
        
        self.num_D = num_D
        
        # Create multiple discriminators
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers)
            setattr(self, f'discriminator_{i}', netD)
        
        # Downsampling layer for creating multi-scale inputs
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
    
    def forward(self, x):
        """
        Returns:
            list of [prediction, features] for each scale
        """
        results = []
        input_downsampled = x
        
        for i in range(self.num_D):
            # Get discriminator for this scale
            netD = getattr(self, f'discriminator_{i}')
            
            # Get prediction and features
            pred, features = netD(input_downsampled)
            results.append([pred, features])
            
            # Downsample for next scale (except for the last one)
            if i != self.num_D - 1:
                input_downsampled = self.downsample(input_downsampled)
        
        return results


class GANLoss(nn.Module):
    """GAN loss (LSGAN)"""
    
    def __init__(self):
        super(GANLoss, self).__init__()
        self.loss = nn.MSELoss()
    
    def __call__(self, prediction, target_is_real):
        """
        Args:
            prediction: discriminator output
            target_is_real: True if target should be real, False if fake
        """
        if target_is_real:
            target = torch.ones_like(prediction)
        else:
            target = torch.zeros_like(prediction)
        
        return self.loss(prediction, target)
    

def get_pix2pixhd_scheduler(optimizer, n_epochs, n_epochs_decay, initial_lr=0.0002):
    """
    Pix2PixHD 공식 Linear Decay Scheduler
    
    Args:
        optimizer: torch optimizer (Generator 또는 Discriminator)
        n_epochs: 총 epoch 수
        n_epochs_decay: decay를 시작할 epoch (보통 n_epochs의 절반)
        initial_lr: 초기 learning rate (default: 0.0002)
    
    Returns:
        LambdaLR scheduler
    """
    def lambda_rule(epoch):
        # epoch이 n_epochs_decay 이전이면 lr 유지 (multiplier = 1.0)
        # 그 이후에는 선형적으로 감소하여 마지막 epoch에 0이 됨
        lr_l = 1.0 - max(0, epoch - n_epochs_decay) / float(n_epochs - n_epochs_decay + 1)
        return lr_l
    
    scheduler = LambdaLR(optimizer, lr_lambda=lambda_rule)
    return scheduler