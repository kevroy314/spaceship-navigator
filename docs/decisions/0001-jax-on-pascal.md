# 0001 — JAX with CUDA 12, and the Pascal kernel gap

**Date** 2026-09-21 · **Status** accepted

## Context
The dev box is a GTX 1080 Ti (Pascal, sm_61, 11 GB) under WSL2. Installing the
current JAX silently fell back to CPU: CUDA 13 wheels dropped Pascal support and
JAX reported no GPU rather than failing loudly.

## Decision
Pin `jax[cuda12]`. Treat "JAX is quietly on CPU" as the first hypothesis
whenever throughput looks wrong.

## Consequences
- Two Pascal-specific costs we pay deliberately:
  - The gradient of `take_along_axis` is a scatter with no Pascal CUBIN, which
    crashed the first PPO update. The loss uses a one-hot product instead.
  - No tensor cores, so fp32 only and no TF32/bf16 speedups.
- `Could not get kernel mode driver version` in the logs is harmless WSL2 noise.
- Moving to Ampere (a 3090 is planned) removes both costs; the one-hot workaround
  can stay, it is cheap.

## Evidence
Measured: 19.4k env steps/s for PPO training, 0.5–1M steps/s for batched
rollout-only work. The 25x gap is observation building plus network
forward/backward, not physics.
