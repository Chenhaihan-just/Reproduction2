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
        description="Generate Table 2 images for Z-Image-Turbo with optional S-NODE"
    )

    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Root output directory, e.g. outputs/table2_baseline or outputs/table2_snode",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="Tongyi-MAI/Z-Image-Turbo",
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
        default=9,
        help="For ZImagePipeline, 9 corresponds to 8 DiT forwards",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=0.0,
        help="Guidance scale for Z-Image-Turbo",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="Base seed for each category. seed = base_seed + image_idx",
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
        "mode": "table2_zimage_snode" if args.snode else "table2_zimage_baseline",
        "model_id": args.model_id,
        "benchmark": args.benchmark,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "base_seed": args.base_seed,
        "attention_backend": args.attention_backend,
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
        "note": "Table 2 generation script for Z-Image-Turbo with optional S-NODE",
    }

    with (output_root / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def init_pipeline(args):
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
    return pipe


def generate_one_image(
    pipe,
    prompt,
    seed,
    height,
    width,
    num_inference_steps,
    guidance_scale,
    snode_pack=None,
    max_sequence_length=512,
):
    if torch.cuda.is_available():
        generator = torch.Generator("cuda").manual_seed(seed)
    else:
        generator = torch.Generator().manual_seed(seed)

    with torch.inference_mode():
        if snode_pack is not None:
            snode_kwargs = snode_pack.pipe_kwargs()

            image = pipe(
                **snode_kwargs,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                max_sequence_length=max_sequence_length,
            ).images[0]
        else:
            image = pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]

    return image


def main():
    args = parse_args()
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

            for image_idx in range(start_idx, end_idx):
                seed = args.base_seed + image_idx
                output_path = category_dir / f"{image_idx:06d}.png"

                if args.resume and output_path.exists():
                    print(f"[Skip] {output_path}")
                    continue

                start_time = time.time()

                image = generate_one_image(
                    pipe=pipe,
                    prompt=prompt,
                    seed=seed,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    snode_pack=snode_pack,
                    max_sequence_length=args.max_sequence_length,
                )

                elapsed = time.time() - start_time

                image.save(output_path)

                print(
                    f"[{benchmark_name}/{category_name}] "
                    f"{image_idx + 1}/{total_count} "
                    f"seed={seed} "
                    f"snode={args.snode} "
                    f"time={elapsed:.2f}s "
                    f"saved={output_path.name}"
                )

                del image
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    total_elapsed = time.time() - total_start
    print(f"\nDone. Total time: {total_elapsed:.2f} seconds")


if __name__ == "__main__":
    main()



