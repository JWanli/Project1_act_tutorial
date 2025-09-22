# Copilot/Agent Instructions for act_tutorial

These notes make AI agents productive immediately in this repo. Focus on concrete workflows, shapes, and local conventions — not generic advice.

## Big Picture
- Purpose: Behavioral cloning of manipulation policies (ACT) in MuJoCo/DM Control simulated tasks; optionally real-world (ALOHA) if installed.
- Key modules:
  - `imitate_episodes.py`: Train/Eval policy over HDF5 episodes.
  - `policy.py`: Policy adapters (`ACTPolicy`, `CNNMLPPolicy`) that wrap models from `detr/`.
  - `utils.py`: Data loading, normalization stats, helpers.
  - `sim_env.py` (joint control) and `ee_sim_env.py` (end-effector control): DM Control envs for tasks; share constants in `constants.py`.
  - `scripted_policy.py`: Scripted EE-space policies to generate demo trajectories.
  - `record_sim_episodes.py`: Roll out scripted policies, replay as joint commands, write episodes to HDF5.
  - `visualize_episodes.py`: Render saved datasets to mp4 + joint plots.

## Data Model (HDF5)
- Per-episode file `episode_{i}.hdf5` with groups/paths:
  - Attr `sim`: bool.
  - `/observations/qpos` and `/observations/qvel`: shape `(T, D_q)`; bimanual tasks typically `D_q=14`.
  - `/observations/images/{camera}`: `(T, 480, 640, 3)` uint8; camera names from task config.
  - `/action`: `(T, D_q)` joint targets (replayed from EE policy with gripper normalization applied).
- Loader: `utils.EpisodicDataset` samples a random `start_ts`, pads actions to episode length, and returns `(image, qpos, action, is_pad)` where:
  - `image`: `(K, C, H, W)` float32 in [0,1], channels-last -> channels-first via einsum; `K=#cameras`.
  - `qpos`: `(D_q,)`, `action`: `(T, D_q)`, `is_pad`: `(T,)` bool.
  - Normalization stats computed across episodes and saved to `ckpt_dir/dataset_stats.pkl`.

## Environments & Tasks
- Task registry: `constants.SIM_TASK_CONFIGS` defines `dataset_dir`, `num_episodes`, `episode_len`, `camera_names` (e.g., `'top'`, `'angle'`).
- Env factories:
  - `sim_env.make_sim_env(task_name)`: joint-space control (14-dim for bimanual). Uses `BOX_POSE[0]` to set object pose on reset.
  - `ee_sim_env.make_ee_sim_env(task_name)`: mocap/EE-space control; scripted policies operate here.
- Scripted generation: rollout in EE space, then replay as joint commands; gripper set via normalize/unnormalize fns in `constants.py`.

## Training & Evaluation
- Models: Built via `detr.main` (`build_ACT_model_and_optimizer`, `build_CNNMLP_model_and_optimizer`). Images normalized with ImageNet stats in `policy.py`.
- Losses:
  - `ACTPolicy`: L1 over predicted action sequences (chunked to `num_queries`), plus KL with weight `kl_weight`.
  - `CNNMLPPolicy`: one-step MSE.
- Temporal aggregation (eval): set `--temporal_agg` to exponentially-weight multiple predicted chunks for step `t`.
- Checkpoints: saved in `ckpt_dir/` (`policy_epoch_*.ckpt`, `policy_best.ckpt`) alongside `dataset_stats.pkl`. Eval expects that stats file in the same folder.

## Common Commands
- Environment setup (prefer the pinned file here):
  ```bash
  conda env create -f conda_env.yaml
  conda activate aloha
  pip install -e ./detr
  ```
- Generate simulated episodes (scripted policy, 50 eps transfer cube):
  ```bash
  python3 record_sim_episodes.py \
    --task_name sim_transfer_cube_scripted \
    --dataset_dir data/sim_transfer_cube_scripted \
    --num_episodes 50
  ```
- Visualize a collected episode:
  ```bash
  python3 visualize_episodes.py --dataset_dir data/sim_transfer_cube_scripted --episode_idx 0
  ```
- Train ACT (example config):
  ```bash
  python3 imitate_episodes.py \
    --task_name sim_transfer_cube_scripted \
    --ckpt_dir ckpt_dir/transfer_cube_all_optimized \
    --policy_class ACT --kl_weight 10 --chunk_size 100 \
    --hidden_dim 512 --dim_feedforward 3200 --nheads 8 \
    --batch_size 8 --num_epochs 2000 --lr 1e-5 --seed 0
  ```
- Evaluate best checkpoint (optional `--temporal_agg`, `--onscreen_render`):
  ```bash
  python3 imitate_episodes.py --eval \
    --task_name sim_transfer_cube_scripted \
    --ckpt_dir ckpt_dir/transfer_cube_all_optimized \
    --policy_class ACT --batch_size 8 --num_epochs 1 --lr 1e-5 --seed 0
  ```

## Project Conventions & Gotchas
- Mutate `sim_env.BOX_POSE[0]` before `env.reset()` to fix object pose for reproducible resets; used in eval and data recording.
- Camera order matters: stack in the order given by `camera_names`; training assumes consistent ordering.
- `val_dataloader` uses `shuffle=True` intentionally to randomize sampled start_ts across epochs.
- Image tensors are normalized twice: first scaled to [0,1] in loaders, then ImageNet Normalize in `policy.py` before forward.
- Action dims match dataset `qpos` dims: e.g., 14 for bimanual tasks, 7 for single-arm A1 pick-and-place.

Questions or gaps? Tell us which sections are unclear (e.g., real-robot setup, task extensions), and I’ll refine this file.
