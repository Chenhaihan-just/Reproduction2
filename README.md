# S-NODE Reproduction

## Overview

This repository contains an unofficial reproduction of **S-NODE**. S-NODE is applied at inference time and requires no model training or modification of the model weights.

We evaluate the method with two open-weight text-to-image models:

| Model | Hugging Face repository | License |
| --- | --- | --- |
| Z-Image-Turbo | [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | [Apache-2.0](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/main/LICENSE) |
| FLUX.2-Klein-4B | [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | [Apache-2.0](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/blob/main/LICENSE.md) |

The experiments cover AFHQ and CelebA-HQ (FID, Recall, and CLIP score), GenEval, and DPG-Bench.

## Released files

Only the inference code used in our experiments and the experimental results are released:

```text
.
├── flux_klein/
│   ├── generate_table2_snode.py
│   ├── generate_geneval_snode.py
│   ├── generate_dpg_snode.py
│   └── snode_full_matrices_False_res_space_slerp.py
├── Z-Image/
│   ├── generate_table2_snode.py
│   ├── generate_geneval_snode.py
│   ├── generate_dpg_snode.py
│   └── snode_full_matrices_False_res_space_slerp.py
└── evaluation_result/
```

The files whose names end in `_snode.py` support both baseline and S-NODE generation. Separate baseline scripts are therefore unnecessary.

All experimental results are provided in `evaluation_result/`. Benchmark datasets, prompts, model weights, and third-party evaluation toolkits are not redistributed; obtain them from their respective official repositories.

## Implementation detail: MDC and the choice of k

In our experiments, the first (largest) singular value is substantially larger than the remaining singular values. Applying Maximum Distance to Chord (MDC) directly to the complete singular-value spectrum therefore selects an excessively small `k`.

To reduce this domination effect, we discard the largest singular value before applying MDC. The selected position is then mapped back to the original spectrum, and the residual-space rotation starts at `k + 2`. This additional offset preserves more leading semantic components and improves semantic alignment between the prompt and the generated image. This is the automatic behavior used when `--snode_fixed_k` is not specified. Passing `--snode_fixed_k K` overrides the automatically selected MDC value.

For reproducibility, S-NODE runs write the MDC details and selected `k` to `snode_mdc_debug.jsonl` in the output directory.

## Generation

The distinction between the two modes is the same for every generation script:

- **Baseline:** run the `_snode.py` script without `--snode`.
- **S-NODE:** run the same command with `--snode --snode_alpha 0.7 --snode_steps 2`.

The defaults used by the scripts are 1024 × 1024 images and seed 42. Z-Image-Turbo uses 9 scheduler steps and guidance 0.0; FLUX.2-Klein-4B uses 4 steps and guidance 1.0. Use different output directories for baseline and S-NODE runs.

### AFHQ and CelebA-HQ

The Table 2 scripts generate the following fixed prompt categories: AFHQ `cat`, `dog`, and `wild`, and CelebA-HQ `man` and `woman`.

**Baseline command:**

```bash
python Z-Image/generate_table2_snode.py \
  --output_root outputs/zimage/table2_baseline \
  --benchmark all \
  --resume
```

**S-NODE command:**

```bash
python Z-Image/generate_table2_snode.py \
  --output_root outputs/zimage/table2_snode \
  --benchmark all \
  --snode \
  --snode_alpha 0.7 \
  --snode_steps 2 \
  --resume
```

For FLUX.2-Klein-4B, use the same commands with `flux_klein/generate_table2_snode.py` and a different output directory:

```bash
# Baseline
python flux_klein/generate_table2_snode.py \
  --output_root outputs/flux_klein/table2_baseline \
  --benchmark all \
  --resume

# S-NODE
python flux_klein/generate_table2_snode.py \
  --output_root outputs/flux_klein/table2_snode \
  --benchmark all \
  --snode \
  --snode_alpha 0.7 \
  --snode_steps 2 \
  --resume
```

Use `--only_category`, `--start_index`, and `--end_index` to split a large run.

### GenEval

Download the official GenEval metadata file before generation.

**Baseline command:**

```bash
python Z-Image/generate_geneval_snode.py \
  --metadata_file <GENEVAL_ROOT>/prompts/evaluation_metadata.jsonl \
  --output_dir outputs/zimage/geneval_baseline \
  --samples_per_prompt 4 \
  --resume
```

**S-NODE command:**

```bash
python Z-Image/generate_geneval_snode.py \
  --metadata_file <GENEVAL_ROOT>/prompts/evaluation_metadata.jsonl \
  --output_dir outputs/zimage/geneval_snode \
  --samples_per_prompt 4 \
  --snode \
  --snode_alpha 0.7 \
  --snode_steps 2 \
  --resume
```

For FLUX.2-Klein-4B, replace the script with `flux_klein/generate_geneval_snode.py` and change the output directory. The generated layout is compatible with GenEval: each prompt directory contains `metadata.jsonl` and a `samples/` directory.

### DPG-Bench

Download the official DPG-Bench prompt files before generation. The standard setting uses four samples per prompt; each output PNG is a 2 × 2 grid.

**Baseline command:**

```bash
python Z-Image/generate_dpg_snode.py \
  --prompts_dir <DPG_BENCH_ROOT>/prompts \
  --output_dir outputs/zimage/dpg_baseline \
  --samples_per_prompt 4 \
  --resume
```

**S-NODE command:**

```bash
python Z-Image/generate_dpg_snode.py \
  --prompts_dir <DPG_BENCH_ROOT>/prompts \
  --output_dir outputs/zimage/dpg_snode \
  --samples_per_prompt 4 \
  --snode \
  --snode_alpha 0.7 \
  --snode_steps 2 \
  --resume
```

For FLUX.2-Klein-4B, replace the script with `flux_klein/generate_dpg_snode.py` and change the output directory.

### Useful options

| Option | Description |
| --- | --- |
| `--snode` | Enables S-NODE; omit it for the baseline. |
| `--snode_alpha` | SLERP steering strength in `[0, 1]` (default: `0.7`). |
| `--snode_steps` | Number of initial denoising steps using the steered embedding (default: `2`). |
| `--snode_fixed_k` | Overrides automatic MDC selection with a fixed `k`. |
| `--snode_svd_device` | Runs SVD on `same`, `cuda`, or `cpu` (default: `same`). |
| `--cpu_offload` | Reduces GPU memory use by enabling model CPU offload. |
| `--resume` | Skips output images that already exist. |
| `--max_prompts` | Limits GenEval or DPG-Bench prompts for debugging. |

Run any script with `--help` for its complete set of options.

## Evaluation

Use the official benchmark implementations to evaluate the generated images:

| Benchmark | Expected generated data | Metrics |
| --- | --- | --- |
| AFHQ / CelebA-HQ | Category directories produced by `generate_table2_snode.py` | FID ↓, Recall ↑, CLIP ↑ |
| GenEval | Prompt directories produced by `generate_geneval_snode.py` | Overall and per-task accuracy ↑ |
| DPG-Bench | 2 × 2 PNG grids produced by `generate_dpg_snode.py` | DPG-Bench score ↑ |

The complete results from our runs are collected under `evaluation_result/`.

## Acknowledgements

This repository builds on PyTorch, Hugging Face Diffusers, Z-Image-Turbo, FLUX.2-Klein-4B, GenEval, and DPG-Bench. Please cite the original S-NODE paper, backbone models, datasets, and benchmark implementations when using this code.

## Disclaimer

This is an unofficial reproduction and is not affiliated with the authors of S-NODE.
