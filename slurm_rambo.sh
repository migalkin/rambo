#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=8gb
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH -p ml
#SBATCH -A p_ml_nlp

#module purge
module load modenv/ml
module load PythonAnaconda/3.7
module load PyTorch
source /projects/p_koop_iais/rambo_env/rambo/bin/activate

python run.py "$@"