#!/bin/bash -l

#SBATCH --job-name=SUNSPOT_TRAIN
#SBATCH --output=/mmfs1/project/wangj/hl545/sunspot/train_outs/%x.%j.out
#SBATCH --error=/mmfs1/project/wangj/hl545/sunspot/train_errs/%x.%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --mem=8000M
#SBATCH --qos=standard
#SBATCH --account=wangj
#SBATCH --time=3-00:00:00

module purge > /dev/null 2>&1
module load wulver # Load slurm, easybuild
conda activate ap
/home/hl545/miniconda3/envs/sunspot/bin/python train.py --config-name wulver
