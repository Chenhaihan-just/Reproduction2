import json
import time
import argparse
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline
from snode_full_matrices_False_res_space_slerp import SNodePromptPack, prepare_snode_prompt_embeds


def parse_args():
    parser = argparse.ArgumentParser(
        description="GenEval baseline inference for FLUX.2-Klein-4B with Diffusers, batch generation"
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
        default="black-forest-labs/FLUX.2-klein-4B",
        help="HF model id or local model path",
    )
    parser.add_argument(
        "--samples_per_prompt",
        type=int,
        default=4,
        help="Number of images to generate per prompt",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for image generation",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="Each prompt uses seeds: base_seed + sample_idx",
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
        default=4,
        help="Number of denoising steps for FLUX.2-Klein",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="Guidance scale for FLUX.2-Klein baseline",
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
        help="End prompt index, not included",
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
        "--cpu_offload",
        action="store_true",
        help="Enable CPU offload instead of pipe.to('cuda')",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile transformer for speed. First run will be slow.",
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
        help="Maximum text sequence length for preparing S-NODE prompt embeddings",
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
        "mode": "geneval_flux2_klein_snode_diffusers_batch"
        if args.snode
        else "geneval_flux2_klein_baseline_diffusers_batch",
        "model_id": args.model_id,
        "metadata_file": args.metadata_file,
        "num_prompts": num_prompts,
        "samples_per_prompt": args.samples_per_prompt,
        "batch_size": args.batch_size,
        "base_seed": args.base_seed,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "compile": args.compile,
        "cpu_offload": args.cpu_offload,
        "snode": args.snode,
        "snode_alpha": args.snode_alpha,
        "snode_steps": args.snode_steps,
        "snode_fixed_k": args.snode_fixed_k,
        "snode_svd_device": args.snode_svd_device,
        "max_sequence_length": args.max_sequence_length,
        "seed_rule": "seed = base_seed + sample_idx",
        "note": "FLUX.2-Klein-4B GenEval batch generation with optional S-NODE.",
    }

    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def make_batched_snode_pack(snode_pack, batch_size: int):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if snode_pack.steered_prompt_embeds.shape[0] == batch_size:
        return snode_pack

    if snode_pack.steered_prompt_embeds.shape[0] != 1:
        raise ValueError(
            "Can only expand an S-NODE prompt pack prepared from one prompt."
        )

    negative_prompt_embeds = None
    if snode_pack.negative_prompt_embeds is not None:
        negative_prompt_embeds = snode_pack.negative_prompt_embeds.repeat(
            batch_size,
            1,
            1,
        ).clone()

    return SNodePromptPack(
        original_prompt_embeds=snode_pack.original_prompt_embeds.repeat(
            batch_size,
            1,
            1,
        ).clone(),
        steered_prompt_embeds=snode_pack.steered_prompt_embeds.repeat(
            batch_size,
            1,
            1,
        ).clone(),
        negative_prompt_embeds=negative_prompt_embeds,
        k=snode_pack.k,
        alpha=snode_pack.alpha,
        num_steering_steps=snode_pack.num_steering_steps,
        prompt_effective_shape=snode_pack.prompt_effective_shape,
        null_effective_shape=snode_pack.null_effective_shape,
        mdc_info=snode_pack.mdc_info,
    )


def main():
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.samples_per_prompt <= 0:
        raise ValueError("--samples_per_prompt must be positive")

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

    print("Loading Flux2KleinPipeline...")
    pipe = Flux2KleinPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
    )

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

    generator_device = "cuda" if torch.cuda.is_available() else "cpu"

    total_images = len(metadata_items) * args.samples_per_prompt
    processed_images = 0
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

        pending_sample_indices = []

        for sample_idx in range(args.samples_per_prompt):
            output_path = samples_dir / f"{sample_idx:04d}.png"

            if args.resume and output_path.exists():
                processed_images += 1
                print(f"[{processed_images}/{total_images}] Skip existing: {output_path}")
            else:
                pending_sample_indices.append(sample_idx)

        if len(pending_sample_indices) == 0:
            continue

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
                    "snode_k": snode_pack.k,
                    "snode_prompt_shape": snode_pack.prompt_effective_shape,
                    "snode_null_shape": snode_pack.null_effective_shape,
                    "mdc": snode_pack.mdc_info,
                }
                with mdc_debug_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(mdc_record, ensure_ascii=False) + "\n")

        for batch_start in range(0, len(pending_sample_indices), args.batch_size):
            batch_sample_indices = pending_sample_indices[
                batch_start: batch_start + args.batch_size
            ]

            batch_prompts = [prompt] * len(batch_sample_indices)
            batch_seeds = [
                args.base_seed + sample_idx for sample_idx in batch_sample_indices
            ]
            batch_generators = [
                torch.Generator(device=generator_device).manual_seed(seed)
                for seed in batch_seeds
            ]

            print(
                f"\nPrompt {prompt_idx:05d}, "
                f"batch samples={batch_sample_indices}, "
                f"batch_size={len(batch_sample_indices)}"
            )
            print(f"Prompt: {prompt}")
            print(f"Seeds: {batch_seeds}")

            batch_time_start = time.time()

            try:
                with torch.inference_mode():
                    if args.snode:
                        batch_snode_pack = make_batched_snode_pack(
                            snode_pack,
                            len(batch_sample_indices),
                        )
                        images = pipe(
                            **batch_snode_pack.pipe_kwargs(),
                            height=args.height,
                            width=args.width,
                            num_inference_steps=args.num_inference_steps,
                            guidance_scale=args.guidance_scale,
                            generator=batch_generators,
                            max_sequence_length=args.max_sequence_length,
                        ).images
                    else:
                        images = pipe(
                            prompt=batch_prompts,
                            height=args.height,
                            width=args.width,
                            num_inference_steps=args.num_inference_steps,
                            guidance_scale=args.guidance_scale,
                            generator=batch_generators,
                        ).images
            except torch.cuda.OutOfMemoryError:
                print("\nCUDA OOM. Try reducing --batch_size to 2 or 1.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise

            batch_elapsed = time.time() - batch_time_start

            if len(images) != len(batch_sample_indices):
                raise RuntimeError(
                    f"Pipeline returned {len(images)} images, "
                    f"but expected {len(batch_sample_indices)}"
                )

            for image, sample_idx, seed in zip(images, batch_sample_indices, batch_seeds):
                output_path = samples_dir / f"{sample_idx:04d}.png"
                image.save(output_path)

                processed_images += 1

                record = {
                    "prompt_index": prompt_idx,
                    "sample_index": sample_idx,
                    "seed": seed,
                    "prompt": prompt,
                    "output_path": str(output_path),
                    "batch_size": len(batch_sample_indices),
                    "batch_elapsed_seconds": batch_elapsed,
                    "estimated_seconds_per_image": batch_elapsed / len(batch_sample_indices),
                    "snode": args.snode,
                }
                if args.snode:
                    record.update(
                        {
                            "snode_alpha": args.snode_alpha,
                            "snode_steps": args.snode_steps,
                            "snode_k": snode_pack.k,
                            "snode_prompt_shape": snode_pack.prompt_effective_shape,
                            "snode_null_shape": snode_pack.null_effective_shape,
                        }
                    )

                with manifest_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

                print(f"[{processed_images}/{total_images}] Saved: {output_path}")

            print(
                f"Batch time: {batch_elapsed:.2f} seconds, "
                f"per image: {batch_elapsed / len(batch_sample_indices):.2f} seconds"
            )

            del images
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    total_elapsed = time.time() - total_start
    print(f"\nDone. Total time: {total_elapsed:.2f} seconds")


if __name__ == "__main__":
    main()






