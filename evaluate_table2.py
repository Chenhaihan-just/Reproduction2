import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from scipy import linalg

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from transformers import CLIPModel, CLIPProcessor


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


TABLE2_PLAN = {
    "afhq": {
        "display_name": "AFHQ",
        "categories": {
            "cat": "A photo of a cat",
            "dog": "A photo of a dog",
            "wild": "A photo of wild animal",
        },
    },
    "celeba_hq": {
        "display_name": "CelebA-HQ",
        "categories": {
            "man": "A photo of a man",
            "woman": "A photo of a woman",
        },
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Table 2 metrics: FID, Recall, and CLIP for AFHQ and CelebA-HQ."
    )

    parser.add_argument(
        "--generated_root",
        type=str,
        required=True,
        help="Generated image root, e.g. outputs/table2_baseline",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Dataset root containing afhq/train and celeba_hq/train",
    )
    parser.add_argument(
        "--real_split",
        type=str,
        default="train",
        help="Real dataset split used as reference. Default: train",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["afhq", "celeba_hq", "all"],
        default="all",
        help="Which benchmark to evaluate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for Inception feature extraction",
    )
    parser.add_argument(
        "--clip_batch_size",
        type=int,
        default=32,
        help="Batch size for CLIPScore computation",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers",
    )
    parser.add_argument(
        "--nearest_k",
        type=int,
        default=3,
        help="k for improved precision/recall manifold estimation",
    )
    parser.add_argument(
        "--distance_chunk_size",
        type=int,
        default=128,
        help="Chunk size for PRDC distance computation",
    )
    parser.add_argument(
        "--max_real",
        type=int,
        default=None,
        help="Optional limit for real images, useful for debugging",
    )
    parser.add_argument(
        "--max_generated",
        type=int,
        default=None,
        help="Optional limit for generated images, useful for debugging",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save metric results as json",
    )

    return parser.parse_args()


def list_images(directory: Path):
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    paths = []
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(path)

    paths = sorted(paths)
    if len(paths) == 0:
        raise ValueError(f"No images found in {directory}")

    return paths


def collect_generated_images(generated_root: Path, benchmark_name: str):
    info = TABLE2_PLAN[benchmark_name]
    generated_items = []

    for category, prompt in info["categories"].items():
        category_dir = generated_root / benchmark_name / category
        paths = list_images(category_dir)

        for path in paths:
            generated_items.append(
                {
                    "path": path,
                    "prompt": prompt,
                    "category": category,
                }
            )

    generated_items = sorted(generated_items, key=lambda x: str(x["path"]))
    return generated_items


def collect_real_images(data_root: Path, benchmark_name: str, split: str):
    real_dir = data_root / benchmark_name / split
    return list_images(real_dir)


class ImagePathDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image


class InceptionFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        weights = Inception_V3_Weights.IMAGENET1K_V1
        self.model = inception_v3(
            weights=weights,
            aux_logits=True,
            transform_input=False,
        )
        self.model.fc = nn.Identity()
        self.model.eval()

    def forward(self, x):
        features = self.model(x)
        if isinstance(features, tuple):
            features = features[0]
        return features


def extract_inception_features(image_paths, batch_size, num_workers, device):
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    dataset = ImagePathDataset(image_paths, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = InceptionFeatureExtractor().to(device)
    model.eval()

    all_features = []

    with torch.inference_mode():
        for images in tqdm(loader, desc="Extracting Inception features"):
            images = images.to(device, non_blocking=True)
            features = model(images)
            all_features.append(features.float().cpu().numpy())

    features = np.concatenate(all_features, axis=0)
    return features


def calculate_fid(real_features, generated_features, eps=1e-6):
    real_features = real_features.astype(np.float64)
    generated_features = generated_features.astype(np.float64)

    mu_real = np.mean(real_features, axis=0)
    mu_gen = np.mean(generated_features, axis=0)

    sigma_real = np.cov(real_features, rowvar=False)
    sigma_gen = np.cov(generated_features, rowvar=False)

    diff = mu_real - mu_gen

    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_gen), disp=False)

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_real.shape[0]) * eps
        covmean = linalg.sqrtm((sigma_real + offset).dot(sigma_gen + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2.0 * covmean)
    return float(fid)


def kth_nearest_neighbor_radii(features, k, chunk_size, device):
    features = torch.from_numpy(features).float().to(device)
    n = features.shape[0]

    radii = []

    with torch.inference_mode():
        for start in tqdm(range(0, n, chunk_size), desc="Computing kNN radii"):
            end = min(start + chunk_size, n)
            chunk = features[start:end]

            distances = torch.cdist(chunk, features, p=2)

            row_indices = torch.arange(end - start, device=device)
            col_indices = torch.arange(start, end, device=device)
            distances[row_indices, col_indices] = float("inf")

            kth = torch.topk(distances, k=k, largest=False, dim=1).values[:, -1]
            radii.append(kth.cpu())

    radii = torch.cat(radii, dim=0).numpy()
    return radii


def calculate_recall(real_features, generated_features, k, chunk_size, device):
    generated_radii = kth_nearest_neighbor_radii(
        generated_features,
        k=k,
        chunk_size=chunk_size,
        device=device,
    )

    real_features_t = torch.from_numpy(real_features).float().to(device)
    generated_features_t = torch.from_numpy(generated_features).float().to(device)
    generated_radii_t = torch.from_numpy(generated_radii).float().to(device)

    n_real = real_features_t.shape[0]
    inside_count = 0

    with torch.inference_mode():
        for start in tqdm(range(0, n_real, chunk_size), desc="Computing Recall"):
            end = min(start + chunk_size, n_real)
            real_chunk = real_features_t[start:end]

            distances = torch.cdist(real_chunk, generated_features_t, p=2)
            inside = (distances <= generated_radii_t.unsqueeze(0)).any(dim=1)

            inside_count += int(inside.sum().item())

    recall = inside_count / n_real
    return float(recall)


def calculate_clip_score(generated_items, batch_size, device):
    model_id = "openai/clip-vit-base-patch32"

    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)

    model.eval()

    scores = []

    for start in tqdm(range(0, len(generated_items), batch_size), desc="Computing CLIP"):
        batch_items = generated_items[start:start + batch_size]

        images = [
            Image.open(item["path"]).convert("RGB")
            for item in batch_items
        ]
        texts = [
            item["prompt"]
            for item in batch_items
        ]

        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            image_features = model.get_image_features(
                pixel_values=inputs["pixel_values"]
            )
            text_features = model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            batch_scores = 100.0 * (image_features * text_features).sum(dim=-1)
            scores.append(batch_scores.cpu().numpy())

    scores = np.concatenate(scores, axis=0)
    return float(scores.mean())


def evaluate_one_benchmark(args, benchmark_name, device):
    generated_root = Path(args.generated_root)
    data_root = Path(args.data_root)

    generated_items = collect_generated_images(generated_root, benchmark_name)
    real_paths = collect_real_images(data_root, benchmark_name, args.real_split)

    if args.max_generated is not None:
        generated_items = generated_items[:args.max_generated]
    if args.max_real is not None:
        real_paths = real_paths[:args.max_real]

    generated_paths = [item["path"] for item in generated_items]

    print("=" * 80)
    print(f"Benchmark       : {TABLE2_PLAN[benchmark_name]['display_name']}")
    print(f"Generated root  : {generated_root / benchmark_name}")
    print(f"Real root       : {data_root / benchmark_name / args.real_split}")
    print(f"Generated images: {len(generated_paths)}")
    print(f"Real images     : {len(real_paths)}")
    print("=" * 80)

    real_features = extract_inception_features(
        real_paths,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    generated_features = extract_inception_features(
        generated_paths,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    fid = calculate_fid(real_features, generated_features)

    recall = calculate_recall(
        real_features=real_features,
        generated_features=generated_features,
        k=args.nearest_k,
        chunk_size=args.distance_chunk_size,
        device=device,
    )

    clip_score = calculate_clip_score(
        generated_items,
        batch_size=args.clip_batch_size,
        device=device,
    )

    result = {
        "FID": fid,
        "Recall": recall,
        "CLIP": clip_score,
        "num_generated": len(generated_paths),
        "num_real": len(real_paths),
    }

    return result


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if args.benchmark == "all":
        selected_benchmarks = ["afhq", "celeba_hq"]
    else:
        selected_benchmarks = [args.benchmark]

    results = {}

    for benchmark_name in selected_benchmarks:
        result = evaluate_one_benchmark(args, benchmark_name, device)
        results[benchmark_name] = result

    print("\nFinal Results")
    print("=" * 80)
    print(f"{'Dataset':<12} {'FID↓':>12} {'Recall↑':>12} {'CLIP↑':>12}")
    print("-" * 80)

    for benchmark_name in selected_benchmarks:
        display_name = TABLE2_PLAN[benchmark_name]["display_name"]
        result = results[benchmark_name]

        print(
            f"{display_name:<12} "
            f"{result['FID']:>12.4f} "
            f"{result['Recall']:>12.4f} "
            f"{result['CLIP']:>12.4f}"
        )

    print("=" * 80)

    if args.output_json is not None:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)

        with output_json.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Saved json results to: {output_json}")


if __name__ == "__main__":
    main()



