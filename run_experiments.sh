#!/bin/bash
source "C:\Users\Admin\miniconda3\etc\profile.d\conda.sh"
conda activate tet
for ((i=0;i<10;i+=1))
do
  for p in TD3 DDPG
  do
    python main.py --policy "$p" --env "HalfCheetah-v5" --seed $i --save_model
    python main.py --policy "$p" --env "Hopper-v5" --seed $i --save_model
    python main.py --policy "$p" --env "Walker2d-v5" --seed $i --save_model
    python main.py --policy "$p" --env "Ant-v5" --seed $i --save_model
    python main.py --policy "$p" --env "Humanoid-v5" --seed $i --save_model

    python main.py --policy "$p" --env "InvertedPendulum-v5" --seed $i --start_timesteps 1000 --save_model
    python main.py --policy "$p" --env "InvertedDoublePendulum-v5" --seed $i --start_timesteps 1000 --save_model
    python main.py --policy "$p" --env "Reacher-v5" --seed $i --start_timesteps 1000 --save_model
  done
done
conda deactivate
echo "TẤT CẢ THỬ NGHIỆM ĐÃ HOÀN THÀNH!"