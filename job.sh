#!/bin/bash
#PBS -l select=1:ncpus=1:mem=128gb -l place=excl
#PBS -l walltime=6:00:00
#PBS -q shortCPUQ

module purge
module load Python/3.9.5-GCCcore-10.3.0

cd $PBS_O_WORKDIR

echo "[+] Installing dependencies..."

#pip install torch torchvision numpy pandas scikit-learn tqdm gdown Pillow

echo "[+] Beginning Vul2Image Algorithm..."

python ./code/process_dataset.py

echo "[+] Finishing processing Dataset..."

python ./code/cnn_pytorch_custom.py

echo "[+] Finishing CNN traning and evaluation."