#!/bin/bash

source /sailhome/kayburns/.bashrc                                                  
# source /sailhome/kayburns/set_cuda_paths.sh                                        
conda deactivate
conda activate py3.7_torch1.8
cd /iris/u/kayburns/continuous-rl/
export PYTHONPATH="$PYTHONPATH:$(pwd)"
cd /iris/u/kayburns/continuous-rl/CQL
export PYTHONPATH="$PYTHONPATH:$(pwd)"
export MJLIB_PATH=/sailhome/kayburns/anaconda3/envs/py3.7_torch1.8/lib/python3.7/site-packages/mujoco_py/binaries/linux/mujoco210/bin/libmujoco210.so
python -m SimpleSAC.conservative_sac_main \
  --env "walker_.01" \
  --logging.output_dir "./debug/" \
  --cql.buffer_file "mix_pendulum" \
  --n_epochs 1 \
  --n_train_step_per_epoch 10 \
  --device 'cuda' 
  # --load_model 'debug/b77da20b7bf844ce9cfb9893a02e754e/'


