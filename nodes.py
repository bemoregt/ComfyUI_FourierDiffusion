"""
ComfyUI FourierDiffusion 커스텀 노드
=====================================
노드 목록:
  1. FourierDiffusionLoader      - 체크포인트에서 모델 로드
  2. FourierDiffusionSampler     - 이미지 생성 (DDPM/DDIM)
  3. FourierDiffusionScheduler   - 노이즈 스케줄 설정
  4. FourierDiffusionInspect     - 주파수 스펙트럼 시각화 (디버깅용)
"""

import os
import torch
import numpy as np
from PIL import Image

import folder_paths  # ComfyUI 내장 모듈

from .model import FourierDiffusionUNet
from .diffusion import FourierDiffusionScheduler


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def _tensor_to_comfy(x: torch.Tensor) -> torch.Tensor:
    """
    모델 출력 [B, C, H, W] (값 범위 [-1, 1])
    → ComfyUI IMAGE 포맷 [B, H, W, C] (값 범위 [0, 1])
    """
    x = (x.clamp(-1.0, 1.0) + 1.0) / 2.0  # [-1,1] → [0,1]
    return x.permute(0, 2, 3, 1).cpu().float()  # [B,C,H,W] → [B,H,W,C]


def _get_device():
    """사용 가능한 최적 장치 반환."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# 1. FourierDiffusionLoader
# ---------------------------------------------------------------------------

class FourierDiffusionLoader:
    """
    FourierDiffusion 체크포인트(.pt / .safetensors)를 로드하고
    추론 준비된 모델 객체를 반환합니다.

    체크포인트 형식:
    {
        "model_state_dict": {...},
        "model_config": {               # 선택적
            "in_channels": 3,
            "model_channels": 64,
            "channel_mults": [1,2,4,8],
            "num_res_blocks": 2,
            "attention_levels": [2,3],
            "dropout": 0.0,
        }
    }
    """

    @classmethod
    def INPUT_TYPES(cls):
        # ComfyUI checkpoints 디렉터리의 파일 목록 사용
        ckpt_files = folder_paths.get_filename_list("checkpoints")
        return {
            "required": {
                "ckpt_name": (ckpt_files,),
            },
            "optional": {
                "model_channels": (
                    "INT",
                    {"default": 64, "min": 16, "max": 512, "step": 16},
                ),
                "channel_mults": (
                    ["1,2,4,8", "1,2,4", "1,2,4,4,8"],
                    {"default": "1,2,4,8"},
                ),
                "num_res_blocks": (
                    "INT",
                    {"default": 2, "min": 1, "max": 4, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("FOURIER_DIFFUSION_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "FourierDiffusion"
    DESCRIPTION = (
        "FourierDiffusion 체크포인트를 로드합니다. "
        "체크포인트에 model_config가 없으면 위 파라미터로 아키텍처를 수동 지정하세요."
    )

    def load(
        self,
        ckpt_name: str,
        model_channels: int = 64,
        channel_mults: str = "1,2,4,8",
        num_res_blocks: int = 2,
    ):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        device = _get_device()

        # 체크포인트 로드
        if ckpt_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state = load_file(ckpt_path, device="cpu")
            model_state = state
            model_cfg = {}
        else:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            if isinstance(ckpt, dict):
                model_state = ckpt.get("model_state_dict", ckpt)
                model_cfg = ckpt.get("model_config", {})
            else:
                raise ValueError(f"지원하지 않는 체크포인트 형식: {type(ckpt)}")

        # 아키텍처 파라미터 결정
        mults = tuple(int(m) for m in channel_mults.split(","))
        cfg = {
            "in_channels": model_cfg.get("in_channels", 3),
            "model_channels": model_cfg.get("model_channels", model_channels),
            "channel_mults": model_cfg.get("channel_mults", mults),
            "num_res_blocks": model_cfg.get("num_res_blocks", num_res_blocks),
            "attention_levels": tuple(model_cfg.get("attention_levels", (2, 3))),
            "dropout": model_cfg.get("dropout", 0.0),
        }

        model = FourierDiffusionUNet(**cfg)
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        if missing:
            print(f"[FourierDiffusion] 누락된 키: {missing[:5]} ...")
        if unexpected:
            print(f"[FourierDiffusion] 예상치 못한 키: {unexpected[:5]} ...")

        model = model.to(device).eval()
        print(f"[FourierDiffusion] 모델 로드 완료 ({device}): {ckpt_name}")
        return (model,)


# ---------------------------------------------------------------------------
# 2. FourierDiffusionScheduler (노드)
# ---------------------------------------------------------------------------

class FourierDiffusionSchedulerNode:
    """
    노이즈 스케줄 설정 노드.
    DDPM/DDIM 샘플러가 사용할 스케줄러 객체를 생성합니다.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timesteps": (
                    "INT",
                    {"default": 1000, "min": 100, "max": 2000, "step": 100},
                ),
                "schedule_type": (
                    ["cosine", "linear"],
                    {"default": "cosine"},
                ),
                "beta_start": (
                    "FLOAT",
                    {"default": 0.0001, "min": 1e-6, "max": 0.01, "step": 0.0001},
                ),
                "beta_end": (
                    "FLOAT",
                    {"default": 0.02, "min": 0.001, "max": 0.5, "step": 0.001},
                ),
            }
        }

    RETURN_TYPES = ("FOURIER_SCHEDULER",)
    RETURN_NAMES = ("scheduler",)
    FUNCTION = "build"
    CATEGORY = "FourierDiffusion"
    DESCRIPTION = "FourierDiffusion 노이즈 스케줄러를 구성합니다."

    def build(
        self,
        timesteps: int,
        schedule_type: str,
        beta_start: float,
        beta_end: float,
    ):
        scheduler = FourierDiffusionScheduler(
            timesteps=timesteps,
            schedule_type=schedule_type,
            beta_start=beta_start,
            beta_end=beta_end,
        )
        return (scheduler,)


# ---------------------------------------------------------------------------
# 3. FourierDiffusionSampler
# ---------------------------------------------------------------------------

class FourierDiffusionSampler:
    """
    FourierDiffusion 이미지 생성 노드.

    모델과 스케줄러를 받아 DDPM 또는 DDIM 방식으로 이미지를 생성합니다.
    스케줄러를 연결하지 않으면 기본 cosine 스케줄(T=1000)을 사용합니다.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("FOURIER_DIFFUSION_MODEL",),
                "width": (
                    "INT",
                    {"default": 256, "min": 64, "max": 1024, "step": 32},
                ),
                "height": (
                    "INT",
                    {"default": 256, "min": 64, "max": 1024, "step": 32},
                ),
                "batch_size": (
                    "INT",
                    {"default": 1, "min": 1, "max": 8, "step": 1},
                ),
                "sampler": (
                    ["ddim", "ddpm"],
                    {"default": "ddim"},
                ),
                "ddim_steps": (
                    "INT",
                    {"default": 50, "min": 10, "max": 500, "step": 10},
                ),
                "eta": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "DDIM eta: 0=결정론적, 1=DDPM과 동일",
                    },
                ),
                "seed": (
                    "INT",
                    {"default": 42, "min": 0, "max": 2**32 - 1},
                ),
                "freq_weight": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 5.0,
                        "step": 0.1,
                        "tooltip": "초기 노이즈 고주파 가중치 (1=균일, >1=고주파 강조)",
                    },
                ),
            },
            "optional": {
                "scheduler": ("FOURIER_SCHEDULER",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "sample"
    CATEGORY = "FourierDiffusion"
    DESCRIPTION = (
        "FourierDiffusion 모델로 이미지를 생성합니다. "
        "DDIM(빠름, 결정론적) 또는 DDPM(느림, 확률적) 샘플러를 선택하세요."
    )

    def sample(
        self,
        model: FourierDiffusionUNet,
        width: int,
        height: int,
        batch_size: int,
        sampler: str,
        ddim_steps: int,
        eta: float,
        seed: int,
        freq_weight: float,
        scheduler=None,
    ):
        device = next(model.parameters()).device
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        # 기본 스케줄러
        if scheduler is None:
            scheduler = FourierDiffusionScheduler(timesteps=1000, schedule_type="cosine")

        in_ch = model.in_channels
        shape = (batch_size, in_ch, height, width)

        # 샘플링
        with torch.no_grad():
            x = scheduler.sample(
                model=model,
                shape=shape,
                device=device,
                sampler=sampler,
                ddim_steps=ddim_steps,
                eta=eta,
                freq_weight=freq_weight,
            )

        images = _tensor_to_comfy(x)
        return (images,)


# ---------------------------------------------------------------------------
# 4. FourierDiffusionInspect
# ---------------------------------------------------------------------------

class FourierDiffusionInspect:
    """
    생성된 이미지의 주파수 스펙트럼을 시각화합니다.
    FourierDiffusion 결과 품질을 디버깅할 때 사용하세요.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "log_scale": ("BOOLEAN", {"default": True}),
                "colormap": (
                    ["inferno", "magma", "viridis", "hot", "gray", "jet"],
                    {"default": "inferno"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("amplitude_spectrum", "phase_spectrum")
    FUNCTION = "inspect"
    CATEGORY = "FourierDiffusion"
    DESCRIPTION = "이미지의 2D FFT 진폭/위상 스펙트럼을 시각화합니다."

    def inspect(self, image: torch.Tensor, log_scale: bool, colormap: str):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        B, H, W, C = image.shape
        amp_outs, phase_outs = [], []

        for b in range(B):
            # 그레이스케일 변환
            gray = image[b].mean(dim=-1).numpy()  # [H, W]

            # FFT
            f = np.fft.fftshift(np.fft.fft2(gray))
            amp = np.abs(f)
            phase = np.angle(f)  # [-π, π]

            if log_scale:
                amp = np.log1p(amp)

            # 정규화 → [0,1]
            def norm01(arr):
                lo, hi = arr.min(), arr.max()
                return (arr - lo) / (hi - lo + 1e-8)

            amp_n = norm01(amp)
            phase_n = (phase + np.pi) / (2 * np.pi)

            cmap_fn = plt.get_cmap(colormap)
            amp_rgb = (cmap_fn(amp_n)[:, :, :3]).astype(np.float32)
            phase_rgb = (cmap_fn(phase_n)[:, :, :3]).astype(np.float32)

            amp_outs.append(torch.from_numpy(amp_rgb))
            phase_outs.append(torch.from_numpy(phase_rgb))

        return (torch.stack(amp_outs), torch.stack(phase_outs))


# ---------------------------------------------------------------------------
# 노드 등록
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "FourierDiffusionLoader": FourierDiffusionLoader,
    "FourierDiffusionScheduler": FourierDiffusionSchedulerNode,
    "FourierDiffusionSampler": FourierDiffusionSampler,
    "FourierDiffusionInspect": FourierDiffusionInspect,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FourierDiffusionLoader": "Fourier Diffusion Loader",
    "FourierDiffusionScheduler": "Fourier Diffusion Scheduler",
    "FourierDiffusionSampler": "Fourier Diffusion Sampler",
    "FourierDiffusionInspect": "Fourier Diffusion Inspect (Spectrum)",
}
