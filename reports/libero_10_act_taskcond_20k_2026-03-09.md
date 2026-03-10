# LIBERO-10 Task-Conditioned ACT (20k Fine-Tune) Report (2026-03-09)

## Summary

- Suite: libero_10 (10 tasks)
- Episodes: 50 per task (500 total)
- Policy: ACT, task-conditioned fine-tune (20k steps) from checkpoint 100000
- Dataset: HuggingFaceVLA/libero (1523 episodes, 246,515 frames)
- Task conditioning: one-hot `observation.environment_state` derived from `task_index`
- Headless rendering: `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`
- GPU: NVIDIA H200 NVL
- Overall success rate: 36.0%

## Training Command

```bash
CUDA_VISIBLE_DEVICES=5 \
lerobot-train \
  --config_path=outputs/train/2026-03-04/11-17-18_act/checkpoints/100000/pretrained_model/train_config.json \
  --output_dir=outputs/train/2026-03-09/act_taskcond \
  --resume=false \
  --policy.pretrained_path=outputs/train/2026-03-04/11-17-18_act/checkpoints/100000/pretrained_model \
  --policy.task_conditioning=true \
  --steps=20000 \
  --save_freq=20000 \
  --eval_freq=0
```

## Evaluation Command

```bash
CUDA_VISIBLE_DEVICES=5 \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
lerobot-eval \
  --policy.path=outputs/train/2026-03-09/act_taskcond/checkpoints/020000/pretrained_model \
  --env.type=libero \
  --env.task=libero_10 \
  --eval.n_episodes=50 \
  --eval.batch_size=10 \
  --output_dir outputs/eval/2026-03-09/libero_10_act_taskcond_h200
```

## Overall Metrics

- Success rate: 36.0%
- Avg sum reward: 0.36
- Avg max reward: 0.36
- Total episodes: 500
- Eval time: 7289.9s (about 2h 1m 30s)

## Per-task Success

| Task ID | Success (count/50) | Success % |
| --- | --- | --- |
| 0 | 0/50 | 0.0% |
| 1 | 4/50 | 8.0% |
| 2 | 31/50 | 62.0% |
| 3 | 43/50 | 86.0% |
| 4 | 7/50 | 14.0% |
| 5 | 26/50 | 52.0% |
| 6 | 20/50 | 40.0% |
| 7 | 8/50 | 16.0% |
| 8 | 13/50 | 26.0% |
| 9 | 28/50 | 56.0% |

## Task Conditioning Details

- Conditioning channel: `observation.environment_state`
- Conditioning type: one-hot task index
- Dataset tasks (total): 40
- LIBERO-10 to dataset task index mapping:
  - `task_id` → `task_index`: `[5, 7, 3, 8, 0, 9, 1, 4, 6, 2]`

## Artifacts

- Training logs: `outputs/train/2026-03-09/act_taskcond/train.log`
- Checkpoint: `outputs/train/2026-03-09/act_taskcond/checkpoints/020000/pretrained_model`
- Metrics: `outputs/eval/2026-03-09/libero_10_act_taskcond_h200/eval_info.json`
- Videos: `outputs/eval/2026-03-09/libero_10_act_taskcond_h200/videos/`
- Video files are renamed to include `_success` or `_fail` in the filename.
