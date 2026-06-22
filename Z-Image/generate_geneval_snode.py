import os
import json
import time
import argparse
from pathlib import Path

import torch
from diffusers import ZImagePipeline
from snode_full_matrices_False_res_space_slerp import prepare_snode_prompt_embeds

def parse_args():
    parser = argparse.ArgumentParser(
        description="GenEval baseline inference for Z-Image-Turbo with Diffusers"
    )

    parser.add_argument(
        "--metadata_file",
        type=str,
        required=True,
        help="Path to GenEval evaluation_metadata.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save GenEval-format outputs",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="Tongyi-MAI/Z-Image-Turbo",
        help="HF model id or local model path",
    )
    parser.add_argument(
        "--samples_per_prompt",
        type=int,
        default=4,
        help="Number of images to generate per prompt",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="Each prompt uses the same seed set: base_seed + sample_idx",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=9,
        help="For ZImagePipeline, 9 corresponds to 8 DiT forwards",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=0.0,
        help="Guidance scale for Z-Image-Turbo baseline",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start prompt index",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="End prompt index (not included)",
    )
    parser.add_argument(
        "--max_prompts",
        type=int,
        default=None,
        help="Optional limit on number of prompts",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip existing images",
    )
    parser.add_argument(
        "--attention_backend",
        type=str,
        default=None,
        choices=[None, "flash", "_flash_3"],
        help="Optional attention backend",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile transformer for speed",
    )
    parser.add_argument(
        "--cpu_offload",
        action="store_true",
        help="Enable CPU offload instead of pipe.to('cuda')",
    )


    parser.add_argument(
        "--snode",
        action="store_true",
        help="Enable S-NODE during inference",
    )
    parser.add_argument(
        "--snode_alpha",
        type=float,
        default=0.7,
        help="S-NODE steering strength alpha",
    )
    parser.add_argument(
        "--snode_steps",
        type=int,
        default=2,
        help="Apply S-NODE for the first N denoising steps",
    )
    parser.add_argument(
        "--snode_fixed_k",
        type=int,
        default=None,
        help="Use fixed k instead of MDC elbow. Default: use MDC",
    )
    parser.add_argument(
        "--snode_svd_device",
        type=str,
        default="same",
        choices=["same", "cuda", "cpu"],
        help="Device for SVD computation",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum text sequence length for ZImagePipeline",
    )
    return parser.parse_args()


def read_metadata(path: str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    items = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            if "prompt" not in item:
                raise KeyError(f"Line {line_idx} missing 'prompt': {item}")
            items.append(item)

    if not items:
        raise ValueError(f"No valid metadata found in {path}")

    return items


def save_run_config(args, output_dir: Path, num_prompts: int):
    cfg = {
        "mode": "geneval_baseline_diffusers",
        "model_id": args.model_id,
        "metadata_file": args.metadata_file,
        "num_prompts": num_prompts,
        "samples_per_prompt": args.samples_per_prompt,
        "base_seed": args.base_seed,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "attention_backend": args.attention_backend,
        "compile": args.compile,
        "cpu_offload": args.cpu_offload,
        "seed_rule": "seed = base_seed + sample_idx",
        "note": "Vanilla Z-Image-Turbo baseline for GenEval",
        "snode": args.snode,
        "snode_alpha": args.snode_alpha,
        "snode_steps": args.snode_steps,
        "snode_fixed_k": args.snode_fixed_k,
        "snode_svd_device": args.snode_svd_device,
        "max_sequence_length": args.max_sequence_length,
    }

    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()

    metadata_items = read_metadata(args.metadata_file)

    if args.max_prompts is not None:
        metadata_items = metadata_items[: args.max_prompts]

    start_index = args.start_index
    end_index = args.end_index if args.end_index is not None else len(metadata_items)
    metadata_items = metadata_items[start_index:end_index]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_run_config(args, output_dir, len(metadata_items))
    manifest_path = output_dir / "manifest.jsonl"
    mdc_debug_path = output_dir / "snode_mdc_debug.jsonl"

    print("Loading ZImagePipeline...")
    pipe = ZImagePipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )

    if args.attention_backend is not None:
        pipe.transformer.set_attention_backend(args.attention_backend)
        print(f"Attention backend: {args.attention_backend}")

    if args.compile:
        pipe.transformer.compile()
        print("Transformer compile enabled")

    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
        print("CPU offload enabled")
    else:
        pipe.to("cuda")
        print("Pipeline moved to cuda")

    pipe.set_progress_bar_config(disable=True)

    total_images = len(metadata_items) * args.samples_per_prompt
    image_counter = 0
    total_start = time.time()

    for local_idx, metadata in enumerate(metadata_items):
        prompt_idx = start_index + local_idx
        prompt = metadata["prompt"]

        prompt_dir = output_dir / f"{prompt_idx:05d}"
        samples_dir = prompt_dir / "samples"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        samples_dir.mkdir(parents=True, exist_ok=True)

        with (prompt_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
            f.write(json.dumps(metadata, ensure_ascii=False) + "\n")


##############################
        snode_pack = None
        if args.snode:
            print(f"[S-NODE] Preparing prompt embeddings for prompt {prompt_idx:05d}")
            snode_pack = prepare_snode_prompt_embeds(
                pipe=pipe,
                prompt=prompt,
                alpha=args.snode_alpha,
                num_steering_steps=args.snode_steps,
                max_sequence_length=args.max_sequence_length,
                fixed_k=args.snode_fixed_k,
                svd_device=args.snode_svd_device,
            )
            print(
                f"[S-NODE] alpha={args.snode_alpha}, "
                f"steps={args.snode_steps}, "
                f"k={snode_pack.k}"
            )
            if snode_pack.mdc_info is not None:
                mdc_record = {
                    "prompt_index": prompt_idx,
                    "prompt": prompt,
                    "snode_alpha": args.snode_alpha,
                    "snode_steps": args.snode_steps,
                    "snode_fixed_k": args.snode_fixed_k,
                    "snode_k": snode_pack.k,
                    "snode_prompt_shape": snode_pack.prompt_effective_shape,
                    "snode_null_shape": snode_pack.null_effective_shape,
                    "mdc": snode_pack.mdc_info,
                }
                with mdc_debug_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(mdc_record, ensure_ascii=False) + "\n")
############################


        for sample_idx in range(args.samples_per_prompt):
            seed = args.base_seed + sample_idx
            output_path = samples_dir / f"{sample_idx:04d}.png"

            image_counter += 1

            if args.resume and output_path.exists():
                print(f"[{image_counter}/{total_images}] Skip existing: {output_path}")
                continue

            print(
                f"\n[{image_counter}/{total_images}] "
                f"Prompt {prompt_idx:05d}, "
                f"Sample {sample_idx + 1}/{args.samples_per_prompt}, "
                f"seed={seed}"
            )
            print(f"Prompt: {prompt}")

            if torch.cuda.is_available():
                generator = torch.Generator("cuda").manual_seed(seed)
            else:
                generator = torch.Generator().manual_seed(seed)

            start = time.time()

            with torch.inference_mode():
                if args.snode:
                    snode_kwargs = snode_pack.pipe_kwargs()

                    image = pipe(
                        **snode_kwargs,
                        height=args.height,
                        width=args.width,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        generator=generator,
                        max_sequence_length=args.max_sequence_length,
                    ).images[0]
                else:
                    image = pipe(
                        prompt=prompt,
                        height=args.height,
                        width=args.width,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        generator=generator,
                        max_sequence_length=args.max_sequence_length,
                    ).images[0]

            elapsed = time.time() - start
            image.save(output_path)

            record = {
                "prompt_index": prompt_idx,
                "sample_index": sample_idx,
                "seed": seed,
                "prompt": prompt,
                "output_path": str(output_path),
                "elapsed_seconds": elapsed,
            }
            if args.snode:
                record.update(
                    {
                        "snode": True,
                        "snode_alpha": args.snode_alpha,
                        "snode_steps": args.snode_steps,
                        "snode_k": snode_pack.k,
                    }
                )
            else:
                record.update({"snode": False})

            with manifest_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(f"Saved: {output_path}")
            print(f"Time taken: {elapsed:.2f} seconds")

            del image
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    total_elapsed = time.time() - total_start
    print(f"\nDone. Total time: {total_elapsed:.2f} seconds")


if __name__ == "__main__":
    main()









