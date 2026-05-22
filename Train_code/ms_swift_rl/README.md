# ms-swift OddGrid GRPO

This directory mirrors the existing EasyR1/verl RL setup, but uses `swift rlhf`
for small 2B smoke tests first.

## Files

- `common.sh`: shared paths, CUDA/env setup, and smoke-test defaults.
- `scripts/prepare_swift_dataset.py`: converts the EasyR1 JSONL format into
  the ms-swift messages/images/solution format.
- `scripts/oddgrid_reward.py`: ms-swift ORM reward plugin with OddGrid EM and F1
  rewards, ported from the EasyR1 reward code.
- `train_qwen35_2b_grpo_smoke.sh`: Qwen3.5-2B GRPO smoke test.
- `train_gemma_2b_grpo_smoke.sh`: Gemma-4-E2B-it GRPO smoke test.
- `train_all_2b_grpo_smoke.sh`: run Qwen3.5-2B and Gemma-4-E2B-it smoke tests serially.

## Run

```bash
cd /nfsdata4/wengtengjin/oddgrid_task/OddGridBench_clean/Train_code/ms_swift_rl
bash train_qwen35_2b_grpo_smoke.sh
bash train_gemma_2b_grpo_smoke.sh
```

The scripts default to a 16-sample train / 8-sample validation smoke run and
`max_steps=5`. Override from the shell when needed:

```bash
TRAIN_LIMIT=128 VAL_LIMIT=32 MAX_STEPS=50 REWARD_FUNC=oddgrid_em bash train_qwen35_2b_grpo_smoke.sh
```

Outputs go to:

```text
/nfsdata4/wengtengjin/oddgrid_task/OddGridBench_clean/Train_code/ms_swift_rl/output
```
