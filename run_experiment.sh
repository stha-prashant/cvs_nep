#!/bin/bash --login

#SBATCH --account yshresth_ai_management
#SBATCH --job-name cvs_nep
#SBATCH --output /scratch/samgain/CVS/logs/%A.out

#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 6
#SBATCH --mem 64G
#SBATCH --time 0-8:00:00
#aSBATCH --array=0-2

module load gcc/11.4.0 cuda/11.8.0 cudnn/8.7.0.84-11.8 python/3.11.6
source /work/FAC/HEC/DESI/yshresth/aim/samgain/CVS/.env/bin/activate

cd /work/FAC/HEC/DESI/yshresth/aim/samgain/CVS/cvs_nep

python3 main.py \
    --pkl_path /scratch/samgain/CVS/ \
    --model resnet50 \
    --num_frames 3 \
    --group 0 \
    --epochs 100 \
    --batch_size 32 \
    --seed 1 \
    --save /scratch/samgain/CVS/ \
    --gpu 0 \
    --lr 5e-6 \
    --reg 0.001 \
    --dropout 0.5 \
    --pretrained \
    --neptune