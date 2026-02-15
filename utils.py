"""
FourierDiffusion 유틸리티
===========================
체크포인트 저장/로드, 이미지 전처리, 학습 루프 헬퍼.
"""

import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path


# ---------------------------------------------------------------------------
# 체크포인트 저장/로드
# ---------------------------------------------------------------------------

def save_checkpoint(
    model,
    path: str,
    scheduler=None,
    optimizer=None,
    epoch: int = 0,
    step: int = 0,
):
    """
    FourierDiffusion 체크포인트 저장.

    저장 형식:
    {
        "model_state_dict": {...},
        "model_config": {...},       # 아키텍처 재현용
        "scheduler_config": {...},   # 스케줄러 파라미터 (선택)
        "optimizer_state_dict": {...},
        "epoch": int,
        "step": int,
    }
    """
    state = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "in_channels": model.in_channels,
            "model_channels": model.model_channels,
            "channel_mults": list(
                [b[0].in_ch if hasattr(b[0], 'in_ch') else model.model_channels
                 for b in model.enc_blocks]
            ),
            "num_res_blocks": model.num_res_blocks,
        },
        "epoch": epoch,
        "step": step,
    }
    if scheduler is not None:
        state["scheduler_config"] = {
            "timesteps": scheduler.timesteps,
        }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"[FourierDiffusion] 체크포인트 저장: {path} (epoch={epoch}, step={step})")


def load_checkpoint(path: str, model, optimizer=None, device=None):
    """
    FourierDiffusion 체크포인트 로드.
    Returns: (epoch, step)
    """
    if device is None:
        device = next(model.parameters()).device

    ckpt = torch.load(path, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(
        ckpt.get("model_state_dict", ckpt), strict=False
    )
    if missing:
        print(f"[FourierDiffusion] 누락된 키 {len(missing)}개: {missing[:3]} ...")
    if unexpected:
        print(f"[FourierDiffusion] 예상치 못한 키 {len(unexpected)}개")

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    epoch = ckpt.get("epoch", 0)
    step = ckpt.get("step", 0)
    print(f"[FourierDiffusion] 체크포인트 로드: {path} (epoch={epoch}, step={step})")
    return epoch, step


# ---------------------------------------------------------------------------
# 이미지 전처리
# ---------------------------------------------------------------------------

def load_image_as_tensor(
    path: str,
    size: tuple = (256, 256),
    normalize: bool = True,
) -> torch.Tensor:
    """
    이미지 파일 → 텐서 [1, C, H, W].

    Args:
        normalize: True → 값 범위 [-1, 1], False → [0, 1]
    """
    img = Image.open(path).convert("RGB").resize(size, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0  # [0, 1]
    if normalize:
        arr = arr * 2.0 - 1.0  # [-1, 1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    return tensor


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    """
    텐서 [1, C, H, W] (값 범위 [-1, 1] 또는 [0, 1]) → PIL Image.
    """
    x = x.squeeze(0).detach().cpu()
    if x.min() < 0:
        x = (x + 1.0) / 2.0
    x = x.clamp(0.0, 1.0)
    arr = (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# 간단한 학습 루프 (데모용)
# ---------------------------------------------------------------------------

def train_step(
    model,
    scheduler,
    batch: torch.Tensor,
    optimizer,
    device,
    loss_in_freq: bool = True,
) -> float:
    """
    단일 배치 학습 스텝.

    Args:
        model         : FourierDiffusionUNet
        scheduler     : FourierDiffusionScheduler
        batch         : 클린 이미지 [B, C, H, W], 범위 [-1, 1]
        optimizer     : torch optimizer
        device        : torch device
        loss_in_freq  : True → 주파수 도메인 MSE, False → 공간 도메인 MSE
    Returns:
        loss (float)
    """
    try:
        from .diffusion import add_fourier_noise
    except ImportError:
        from diffusion import add_fourier_noise

    model.train()
    batch = batch.to(device)
    B = batch.shape[0]

    # 랜덤 타임스텝 샘플링
    t = torch.randint(0, scheduler.timesteps, (B,), device=device)

    # 노이즈 추가
    noisy_list, noise_list = [], []
    for i in range(B):
        noisy, noise = add_fourier_noise(
            batch[i:i+1], t[i].item(),
            scheduler.sqrt_alphas_cumprod.to(device),
            scheduler.sqrt_one_minus_alphas_cumprod.to(device),
        )
        noisy_list.append(noisy)
        noise_list.append(noise)
    noisy_batch = torch.cat(noisy_list, dim=0)
    noise_batch = torch.cat(noise_list, dim=0)

    # 노이즈 예측
    pred_noise = model(noisy_batch, t)

    # 손실 계산
    if loss_in_freq:
        # 주파수 도메인 MSE (저주파/고주파 균형)
        pred_f = torch.fft.rfft2(pred_noise, norm="ortho")
        true_f = torch.fft.rfft2(noise_batch, norm="ortho")
        loss = (
            (pred_f.real - true_f.real) ** 2 +
            (pred_f.imag - true_f.imag) ** 2
        ).mean()
    else:
        loss = torch.nn.functional.mse_loss(pred_noise, noise_batch)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss.item()
