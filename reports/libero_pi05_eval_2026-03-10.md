# LIBERO Pi05 Evaluation Report (2026-03-10)

## Summary

- Suites: libero_spatial, libero_object, libero_goal, libero_10 (10 episodes per task)
- Policy: Pi05, `lerobot/pi05_libero_finetuned`
- Headless rendering: `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`
- GPU: NVIDIA H200 NVL
- Overall success rate: 97.5% (400 episodes total)

## Command

```bash
CUDA_VISIBLE_DEVICES=5 \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
lerobot-eval \
  --policy.path=lerobot/pi05_libero_finetuned \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --eval.n_episodes=10 \
  --eval.batch_size=1 \
  --policy.n_action_steps=10 \
  --env.max_parallel_tasks=1 \
  --output_dir outputs/eval/2026-03-10/pi05_libero_finetuned_h200
```

## Overall Metrics

- Success rate: 97.5%
- Avg sum reward: 0.975
- Avg max reward: 0.975
- Total episodes: 400
- Eval time: 6287.9s (about 1h 44m 48s)

## Per-suite Success

| Suite | Success % | Episodes |
| --- | --- | --- |
| libero_spatial | 98.0% | 100 |
| libero_object | 100.0% | 100 |
| libero_goal | 99.0% | 100 |
| libero_10 | 93.0% | 100 |

## Artifacts

- Metrics: `outputs/eval/2026-03-10/pi05_libero_finetuned_h200/eval_info.json`
- Videos: `outputs/eval/2026-03-10/pi05_libero_finetuned_h200/videos/`
