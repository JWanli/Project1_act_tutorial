python imitate_episodes.py \
    --task_name sim_pick_n_place_cube_scripted \
    --ckpt_dir ./ckpt_dir/pick2 \
    --policy_class ACT \
    --kl_weight 5 \
    --chunk_size 200 \
    --hidden_dim 1024 \
    --nheads 16 \
    --batch_size 6 \
    --dim_feedforward 4096 \
    --num_epochs 2000 \
    --lr 1e-5 \
    --seed 0 \
    --backbone resnet50 \
    --l1_weight 1.0 \
    --recon_loss huber --huber_beta 0.1 \
    --adam_beta1 0.9 --adam_beta2 0.98 \
    # --eval \