from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F



def _slerp_residual_block(v0, v1, alpha: float, eps: float = 1e-6):
    """Apply one SLERP to the whole residual block flattened as a vector."""
    original_dtype = v0.dtype
    original_shape = v0.shape

    v0 = v0.float()
    v1 = v1.float()

    # Resolve per-vector SVD sign ambiguity before measuring block direction.
    row_dot = (v0 * v1).sum(dim=-1, keepdim=True)
    v1 = torch.where(row_dot < 0, -v1, v1)

    v0_flat = v0.reshape(1, -1)
    v1_flat = v1.reshape(1, -1)

    v0_norm = F.normalize(v0_flat, dim=-1)
    v1_norm = F.normalize(v1_flat, dim=-1)

    dot = (v0_norm * v1_norm).sum(dim=-1, keepdim=True).clamp(
        -1.0 + eps,
        1.0 - eps,
    )
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    s0 = torch.sin((1.0 - alpha) * theta) / sin_theta
    s1 = torch.sin(alpha * theta) / sin_theta
    out = s0 * v0_flat + s1 * v1_flat

    lerp = (1.0 - alpha) * v0_flat + alpha * v1_flat
    out = torch.where(sin_theta.abs() < eps, lerp, out)

    return out.reshape(original_shape).to(original_dtype)



def _mdc_elbow_info(
    singular_values,
    min_k: int = 1,
    max_k: Optional[int] = None,
):
    """Return the MDC calculation details and selected component count."""
    s = singular_values.detach().float()
    r = s.numel()

    if r <= 2:
        x = torch.linspace(0, 1, r, device=s.device) if r > 0 else s
        return {
            "rank": int(r),
            "min_k": int(min_k),
            "max_k": None if max_k is None else int(max_k),
            "singular_values": s.cpu().tolist(),
            "normalized_singular_values": [0.0 for _ in range(r)],
            "x": x.cpu().tolist(),
            "distances": [0.0 for _ in range(r)],
            "masked_distances": [0.0 for _ in range(r)],
            "selected_index": 0,
            "selected_k": 1,
        }

    if max_k is None:
        max_k = r - 1

    original_min_k = min_k
    original_max_k = max_k
    min_k = max(0, min_k)
    max_k = min(max_k, r - 1)

    y = (s - s.min()) / (s.max() - s.min() + 1e-8)
    x = torch.linspace(0, 1, r, device=s.device)
    points = torch.stack([x, y], dim=1)

    a = points[0]
    b = points[-1]
    ab = b - a

    ap = points - a
    dist = torch.abs(ab[0] * ap[:, 1] - ab[1] * ap[:, 0]) / (torch.norm(ab) + 1e-8)
    raw_dist = dist.clone()

    # k is a component count, so singular-value index i corresponds to k=i+1.
    if min_k > 0:
        dist[: min_k - 1] = -1
    dist[max_k:] = -1

    idx = int(torch.argmax(dist).item())
    return {
        "rank": int(r),
        "min_k": int(original_min_k),
        "max_k": int(original_max_k),
        "effective_min_k": int(min_k),
        "effective_max_k": int(max_k),
        "singular_values": s.cpu().tolist(),
        "normalized_singular_values": y.cpu().tolist(),
        "x": x.cpu().tolist(),
        "distances": raw_dist.cpu().tolist(),
        "masked_distances": dist.cpu().tolist(),
        "selected_index": idx,
        "selected_k": idx + 1,
    }


def _mdc_elbow_info_skip_first(
    singular_values,
    min_k: int = 1,
    max_k: Optional[int] = None,
):
    """Run MDC on S[1:] and map the result back to the original rank."""
    s = singular_values.detach().float()
    r = s.numel()

    if r <= 2:
        info = _mdc_elbow_info(s, min_k=min_k, max_k=max_k)
        info["skip_first"] = False
        info["skip_reason"] = "rank <= 2"
        return info

    shifted_min_k = max(1, min_k - 1)
    shifted_max_k = None if max_k is None else max(1, max_k - 1)
    shifted_info = _mdc_elbow_info(
        s[1:],
        min_k=shifted_min_k,
        max_k=shifted_max_k,
    )

    selected_k_without_first = shifted_info["selected_k"]
    selected_k = selected_k_without_first + 1

    return {
        **shifted_info,
        "skip_first": True,
        "original_rank": int(r),
        "original_singular_values": s.cpu().tolist(),
        "selected_k_without_first": int(selected_k_without_first),
        "selected_index_without_first": int(shifted_info["selected_index"]),
        "selected_index": int(shifted_info["selected_index"] + 1),
        "selected_k": int(selected_k),
    }


@dataclass
class SNodePromptPack:
    original_prompt_embeds: list
    steered_prompt_embeds: list
    negative_prompt_embeds: Optional[list]
    k: int
    alpha: float
    num_steering_steps: int
    prompt_effective_shape: Optional[tuple[int, int]] = None
    null_effective_shape: Optional[tuple[int, int]] = None
    mdc_info: Optional[dict] = None

    def callback(self):
        original_prompt_embeds = self.original_prompt_embeds
        num_steering_steps = self.num_steering_steps

        def _callback(pipe, step_index, timestep, callback_kwargs):

            if step_index >= num_steering_steps - 1:
                callback_kwargs["prompt_embeds"] = original_prompt_embeds
            return callback_kwargs

        return _callback

    def pipe_kwargs(self):
        kwargs = {
            "prompt": None,
            "prompt_embeds": self.steered_prompt_embeds,
            "callback_on_step_end": self.callback(),
            "callback_on_step_end_tensor_inputs": ["latents", "prompt_embeds"],
        }

        if self.negative_prompt_embeds is not None:
            kwargs["negative_prompt_embeds"] = self.negative_prompt_embeds

        return kwargs


@torch.no_grad()
def prepare_snode_prompt_embeds(
    pipe,
    prompt: str,
    alpha: float = 0.7,
    num_steering_steps: int = 2,
    max_sequence_length: int = 512,
    fixed_k: Optional[int] = None,
    min_k: int = 1,
    max_k: Optional[int] = None,
    svd_device: str = "same",
):
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if num_steering_steps < 1:
        raise ValueError(
            f"num_steering_steps must be at least 1, got {num_steering_steps}"
        )

    device = pipe._execution_device

    prompt_embeds, _ = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        do_classifier_free_guidance=False,
        max_sequence_length=max_sequence_length,
    )

    null_embeds, _ = pipe.encode_prompt(
        prompt="",
        device=device,
        do_classifier_free_guidance=False,
        max_sequence_length=max_sequence_length,
    )

    c = prompt_embeds[0]
    c_null = null_embeds[0]

    original_device = c.device
    original_dtype = c.dtype

    if svd_device == "cpu":
        svd_target_device = torch.device("cpu")
    elif svd_device == "cuda":
        svd_target_device = torch.device("cuda")
    else:
        svd_target_device = c.device

    c_svd = c.to(device=svd_target_device, dtype=torch.float32)
    c_null_svd = c_null.to(device=svd_target_device, dtype=torch.float32)

    try:
        U, S, Vh = torch.linalg.svd(c_svd, full_matrices=False)
        _, S_null, Vh_null = torch.linalg.svd(c_null_svd, full_matrices=False)
    except RuntimeError as e:
        if svd_device != "cpu":
            print("[S-NODE] CUDA SVD failed. Retrying on CPU.")
            c_svd = c.cpu().float()
            c_null_svd = c_null.cpu().float()
            U, S, Vh = torch.linalg.svd(c_svd, full_matrices=False)
            _, S_null, Vh_null = torch.linalg.svd(
                c_null_svd, full_matrices=False
            )
        else:
            raise e

    r = S.numel()
    null_r = S_null.numel()

    mdc_info = _mdc_elbow_info_skip_first(
        S,
        min_k=min_k,
        max_k=max_k,
    )
    if fixed_k is None:
        k = mdc_info["selected_k"]
    else:
        k = int(fixed_k)
        mdc_info["fixed_k"] = k
        mdc_info["selected_k_before_fixed"] = mdc_info["selected_k"]
        mdc_info["selected_k"] = k

    # Keep at least one prompt residual direction whenever its rank allows it.
    k = max(0, min(k, r - 1))
    mdc_info["clamped_k"] = int(k)

    Vh_hat = Vh.clone()

    shared_rank = min(r, null_r)
    rotate_start = min(shared_rank, k + 2)
    mdc_info["null_rank"] = int(null_r)
    mdc_info["shared_rank"] = int(shared_rank)
    mdc_info["rotate_start"] = int(rotate_start)
    mdc_info["rotated_components"] = int(max(0, shared_rank - rotate_start))
    if rotate_start < shared_rank:
        source_res = Vh[rotate_start:shared_rank, :]
        target_res = Vh_null[rotate_start:shared_rank, :].to(source_res.device)

        Vh_hat[rotate_start:shared_rank, :] = _slerp_residual_block(
            source_res,
            target_res,
            alpha=alpha,
        )

    c_hat = (U * S.unsqueeze(0)) @ Vh_hat
    c_hat = c_hat.to(device=original_device, dtype=original_dtype)

    original_prompt_embeds = [
        c.to(device=original_device, dtype=original_dtype).clone()
    ]
    steered_prompt_embeds = [c_hat.clone()]

    # Only needed when guidance_scale > 0, but safe to prepare here.
    negative_prompt_embeds = [
        c_null.to(device=original_device, dtype=original_dtype).clone()
    ]

    return SNodePromptPack(
        original_prompt_embeds=original_prompt_embeds,
        steered_prompt_embeds=steered_prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        k=k,
        alpha=alpha,
        num_steering_steps=num_steering_steps,
        prompt_effective_shape=tuple(c.shape),
        null_effective_shape=tuple(c_null.shape),
        mdc_info=mdc_info,
    )
