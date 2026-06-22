import json
import time
import argparse
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline
from snode_full_matrices_False_res_space_slerp import (
    SNodePromptPack,
    prepare_snode_prompt_embeds,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Table 2 images for FLUX.2-Klein-4B"
    )

    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Root output directory, e.g. outputs/table2_flux2klein_baseline",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="black-forest-labs/FLUX.2-klein-4B",
        help="HF model id or local model path",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["afhq", "celeba_hq", "all"],
        default="all",
        help="Which benchmark to generate",
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
        help="Guidance scale for FLUX.2-Klein",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="Base seed for each category. seed = base_seed + image_idx",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip existing images",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile transformer for speed. First run will be slow.",
    )
    parser.add_argument(
        "--cpu_offload",
        action="store_true",
        help="Enable CPU offload",
    )
    parser.add_argument(
        "--only_category",
        type=str,
        default=None,
        help="Generate only one category, e.g. cat / dog / wild / man / woman",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start image index for resume/splitting",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="End image index, not included",
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


def build_table2_plan():
    plan = {
        "afhq": {
            "cat": {
                "prompt": "A photo of a cat",
                "count": 5000,
            },
            "dog": {
                "prompt": "A photo of a dog",
                "count": 5000,
            },
            "wild": {
                "prompt": "A photo of wild animal",
                "count": 5000,
            },
        },
        "celeba_hq": {
            "man": {
                "prompt": "A photo of a man",
                "count": 10057,
            },
            "woman": {
                "prompt": "A photo of a woman",
                "count": 17943,
            },
        },
    }
    return plan


def save_config(args, output_root: Path, plan):
    cfg = {
        "mode": "table2_flux2klein_snode_diffusers_batch"
        if args.snode
        else "table2_flux2klein_baseline_diffusers_batch",
        "model_id": args.model_id,
        "benchmark": args.benchmark,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "base_seed": args.base_seed,
        "batch_size": args.batch_size,
        "compile": args.compile,
        "cpu_offload": args.cpu_offload,
        "resume": args.resume,
        "only_category": args.only_category,
        "start_index": args.start_index,
        "end_index": args.end_index,
        "snode": args.snode,
        "snode_alpha": args.snode_alpha,
        "snode_steps": args.snode_steps,
        "snode_fixed_k": args.snode_fixed_k,
        "snode_svd_device": args.snode_svd_device,
        "max_sequence_length": args.max_sequence_length,
        "seed_rule": "seed = base_seed + image_idx",
        "plan": plan,
        "note": "Table 2 generation script for FLUX.2-Klein-4B with optional S-NODE",
    }

    with (output_root / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def init_pipeline(args):
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
    return pipe


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


def generate_batch(
    pipe,
    prompt,
    seeds,
    height,
    width,
    num_inference_steps,
    guidance_scale,
    snode_pack=None,
    max_sequence_length=512,
):
    generator_device = "cuda" if torch.cuda.is_available() else "cpu"

    prompts = [prompt] * len(seeds)
    generators = [
        torch.Generator(device=generator_device).manual_seed(seed)
        for seed in seeds
    ]

    with torch.inference_mode():
        if snode_pack is not None:
            batch_snode_pack = make_batched_snode_pack(snode_pack, len(seeds))
            images = pipe(
                **batch_snode_pack.pipe_kwargs(),
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generators,
                max_sequence_length=max_sequence_length,
            ).images
        else:
            images = pipe(
                prompt=prompts,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generators,
            ).images

    return images


def main():
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")

    plan = build_table2_plan()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    mdc_debug_path = output_root / "snode_mdc_debug.jsonl"

    save_config(args, output_root, plan)

    pipe = init_pipeline(args)

    if args.benchmark == "all":
        selected_benchmarks = ["afhq", "celeba_hq"]
    else:
        selected_benchmarks = [args.benchmark]

    total_start = time.time()

    for benchmark_name in selected_benchmarks:
        categories = plan[benchmark_name]

        for category_name, info in categories.items():
            if args.only_category is not None and category_name != args.only_category:
                continue

            prompt = info["prompt"]
            total_count = info["count"]

            start_idx = args.start_index
            end_idx = args.end_index if args.end_index is not None else total_count
            end_idx = min(end_idx, total_count)

            category_dir = output_root / benchmark_name / category_name
            category_dir.mkdir(parents=True, exist_ok=True)

            meta = {
                "benchmark": benchmark_name,
                "category": category_name,
                "prompt": prompt,
                "count": total_count,
                "model_id": args.model_id,
                "num_inference_steps": args.num_inference_steps,
                "guidance_scale": args.guidance_scale,
                "snode": args.snode,
                "snode_alpha": args.snode_alpha if args.snode else None,
                "snode_steps": args.snode_steps if args.snode else None,
                "seed_rule": "seed = base_seed + image_idx",
            }

            with (category_dir / "metadata.json").open("w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            print("=" * 80)
            print(f"Benchmark : {benchmark_name}")
            print(f"Category  : {category_name}")
            print(f"Prompt    : {prompt}")
            print(f"Range     : [{start_idx}, {end_idx}) / {total_count}")
            print(f"Batch size: {args.batch_size}")
            print(f"S-NODE    : {args.snode}")
            print("=" * 80)

            snode_pack = None
            if args.snode:
                print(f"[S-NODE] Preparing prompt embeddings for {benchmark_name}/{category_name}")
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

                meta["snode_k"] = snode_pack.k
                meta["snode_fixed_k"] = args.snode_fixed_k
                meta["snode_prompt_shape"] = snode_pack.prompt_effective_shape
                meta["snode_null_shape"] = snode_pack.null_effective_shape
                with (category_dir / "metadata.json").open("w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2, ensure_ascii=False)

                if snode_pack.mdc_info is not None:
                    mdc_record = {
                        "benchmark": benchmark_name,
                        "category": category_name,
                        "prompt": prompt,
                        "range_start": start_idx,
                        "range_end": end_idx,
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

            pending_indices = []

            for image_idx in range(start_idx, end_idx):
                output_path = category_dir / f"{image_idx:06d}.png"

                if args.resume and output_path.exists():
                    print(f"[Skip] {output_path}")
                    continue

                pending_indices.append(image_idx)

            if len(pending_indices) == 0:
                print(f"No pending images for {benchmark_name}/{category_name}")
                continue

            for batch_start in range(0, len(pending_indices), args.batch_size):
                batch_indices = pending_indices[
                    batch_start: batch_start + args.batch_size
                ]

                seeds = [args.base_seed + image_idx for image_idx in batch_indices]

                print(
                    f"[{benchmark_name}/{category_name}] "
                    f"Generating indices={batch_indices}, seeds={seeds}"
                )

                start_time = time.time()

                try:
                    images = generate_batch(
                        pipe=pipe,
                        prompt=prompt,
                        seeds=seeds,
                        height=args.height,
                        width=args.width,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        snode_pack=snode_pack,
                        max_sequence_length=args.max_sequence_length,
                    )
                except torch.cuda.OutOfMemoryError:
                    print("\nCUDA OOM. Try reducing --batch_size to 2 or 1.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise

                elapsed = time.time() - start_time

                if len(images) != len(batch_indices):
                    raise RuntimeError(
                        f"Pipeline returned {len(images)} images, "
                        f"but expected {len(batch_indices)}"
                    )

                for image, image_idx, seed in zip(images, batch_indices, seeds):
                    output_path = category_dir / f"{image_idx:06d}.png"
                    image.save(output_path)

                    print(
                        f"  saved={output_path.name} "
                        f"seed={seed}"
                    )

                print(
                    f"Batch time={elapsed:.2f}s, "
                    f"per image={elapsed / len(batch_indices):.2f}s"
                )

                del images

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    total_elapsed = time.time() - total_start
    print(f"\nDone. Total time: {total_elapsed:.2f} seconds")


if __name__ == "__main__":
    main()




