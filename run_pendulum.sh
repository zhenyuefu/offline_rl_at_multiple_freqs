#!/bin/bash
#SBATCH --partition=iris-hi
#SBATCH --exclude=iris-hp-z8,iris1,iris4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name="cql"
#SBATCH --time=3-0:0

#source /sailhome/kayburns/.bashrc
# source /sailhome/kayburns/set_cuda_paths.sh                                        
python -m SimpleSAC.conservative_sac_main \
  --env "pendulum" \
  --logging.output_dir "./rebuttal/pendulum/" \
  --logging.online True \
  --logging.project 'rebuttal' \
  --cql.cql_min_q_weight 5 \
  --cql.policy_lr 3e-4 \
  --cql.qf_lr 3e-4 \
  --cql.discount .99 \
  --n_epochs 500 \
  --seed ${1} \
  --N_steps ${2} \
  --device 'cuda' \
  --save_model True
