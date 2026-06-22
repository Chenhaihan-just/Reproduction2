import json
import time
import argparse
from pathlib import Path

import torch
from PIL import Image
from diffusers import Flux2KleinPipeline
from snode_full_matrices_False_res_space_slerp import (
    SNodePromptPack,
    prepare_snode_prompt_embeds,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="DPG-Bench baseline inference for FLUX.2-Klein-4B with Diffusers"
    )

    parser.add_argument(
        "--prompts_dir",
        type=str,
        required=True,
        help="Path to DPG-Bench prompts directory, e.g. dpg_bench/prompts",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save DPG grid images. This directory should contain only png images for evaluation.",
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
        help="DPG-Bench recommends 4 images per prompt",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for generation. For DPG, use 4 by default.",
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
        help="Single image height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Single image width",
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
        "--max_prompts",
        type=int,
        default=None,
        help="Optional limit for debugging",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start index after sorting prompt files",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="End index after sorting prompt files, not included",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip existing grid images",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile transformer for speed. First run will be slow.",
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
        help="Maximum text sequence length for preparing S-NODE prompt embeddings",
    )

    return parser.parse_args()


def numeric_sort_key(path: Path):
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.stem)


def read_prompt_file(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty prompt file: {path}")
    return text


def make_2x2_grid(images, width: int, height: int):
    if len(images) != 4:
        raise ValueError("DPG 2x2 grid requires exactly 4 images.")

    grid = Image.new("RGB", (width * 2, height * 2))

    positions = [
        (0, 0),
        (width, 0),
        (0, height),
        (width, height),
    ]

    for image, pos in zip(images, positions):
        if image.size != (width, height):
            image = image.resize((width, height), Image.BICUBIC)
        grid.paste(image.convert("RGB"), pos)

    return grid


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

    prompts_dir = Path(args.prompts_dir)
    output_dir = Path(args.output_dir)

    if not prompts_dir.exists():
        raise FileNotFoundError(f"Prompts directory not found: {prompts_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    mdc_debug_path = output_dir / "snode_mdc_debug.jsonl"

    prompt_files = sorted(prompts_dir.glob("*.txt"), key=numeric_sort_key)

    if args.max_prompts is not None:
        prompt_files = prompt_files[: args.max_prompts]

    start_index = args.start_index
    end_index = args.end_index if args.end_index is not None else len(prompt_files)
    prompt_files = prompt_files[start_index:end_index]

    if not prompt_files:
        raise ValueError(f"No prompt files found in {prompts_dir}")

    if args.samples_per_prompt != 4:
        raise ValueError(
            "DPG-Bench standard setting uses 4 images per prompt and a 2x2 grid. "
            "Please keep --samples_per_prompt 4."
        )

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")

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

    total_start = time.time()

    for file_idx, prompt_file in enumerate(prompt_files):
        prompt_id = prompt_file.stem
        prompt = read_prompt_file(prompt_file)

        output_path = output_dir / f"{prompt_id}.png"

        if args.resume and output_path.exists():
            print(
                f"[{file_idx + 1}/{len(prompt_files)}] "
                f"Skip existing: {output_path}"
            )
            continue

        print(
            f"\n[{file_idx + 1}/{len(prompt_files)}] "
            f"Prompt file: {prompt_file.name}"
        )
        print(f"Prompt: {prompt}")

        snode_pack = None
        if args.snode:
            print(f"[S-NODE] Preparing prompt embeddings for {prompt_file.name}")
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
                    "prompt_index": start_index + file_idx,
                    "prompt_id": prompt_id,
                    "prompt_file": str(prompt_file),
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

        all_images = []

        sample_indices = list(range(args.samples_per_prompt))

        for batch_start in range(0, args.samples_per_prompt, args.batch_size):
            batch_sample_indices = sample_indices[
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
                f"  Generating batch samples={batch_sample_indices}, "
                f"seeds={batch_seeds}, "
                f"batch_size={len(batch_sample_indices)}"
            )

            start = time.time()

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

            elapsed = time.time() - start
            print(
                f"  Batch time: {elapsed:.2f}s, "
                f"per image: {elapsed / len(batch_sample_indices):.2f}s"
            )

            if len(images) != len(batch_sample_indices):
                raise RuntimeError(
                    f"Pipeline returned {len(images)} images, "
                    f"but expected {len(batch_sample_indices)}"
                )

            all_images.extend(images)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        grid = make_2x2_grid(all_images, width=args.width, height=args.height)
        grid.save(output_path)

        print(f"Saved DPG grid: {output_path}")

        del all_images
        del grid

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_elapsed = time.time() - total_start
    print(f"\nDone. Total time: {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()


