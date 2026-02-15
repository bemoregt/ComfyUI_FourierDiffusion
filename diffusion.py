"""
FourierDiffusion - 확산 프로세스
===================================
DDPM/DDIM 스타일 노이즈 스케줄과 샘플링 루프.
주파수 도메인에서 노이즈를 추가/제거하는 것이 핵심.

주요 특징:
 - 순방향(forward): FFT 계수에 가우시안 노이즈 추가
 - 역방향(reverse): UNet이 예측한 노이즈를 빼며 복원
 - DDIM 결정론적 샘플링 지원
"""

import math
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 노이즈 스케줄
# ---------------------------------------------------------------------------

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Improved DDPM cosine schedule (Nichol & Dhariwal, 2021)."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0.0001, 0.9999)


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """선형 베타 스케줄."""
    return torch.linspace(beta_start, beta_end, timesteps)


# ---------------------------------------------------------------------------
# 주파수 도메인 노이즈 유틸
# ---------------------------------------------------------------------------

def add_fourier_noise(
    x: torch.Tensor,
    t_idx: int,
    sqrt_alphas_cumprod: torch.Tensor,
    sqrt_one_minus_alphas_cumprod: torch.Tensor,
    freq_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    주파수 도메인에서 노이즈 추가 (DDPM q 분포).

    공간 도메인이 아닌 FFT 계수에 노이즈를 추가하므로
    저주파와 고주파 성분을 분리하여 처리 가능.

    Args:
        x             : 클린 이미지 [B, C, H, W], 값 범위 [-1, 1]
        t_idx         : 타임스텝 인덱스
        freq_weight   : 고주파 성분 노이즈 가중치 (1.0 = 균일)
    Returns:
        (noisy_x, noise): 노이즈가 추가된 이미지와 순수 노이즈
    """
    B, C, H, W = x.shape
    device = x.device

    # FFT 도메인으로 변환
    X_freq = torch.fft.rfft2(x, norm="ortho")

    # 주파수 가중 노이즈 생성
    noise_real = torch.randn_like(X_freq.real)
    noise_imag = torch.randn_like(X_freq.imag)

    if freq_weight != 1.0:
        # 고주파 노이즈 가중치 적용
        fH, fW = X_freq.shape[-2], X_freq.shape[-1]
        fy = torch.fft.fftfreq(H, device=device)[:fH].abs()
        fx = torch.fft.rfftfreq(W, device=device)[:fW]
        freq_mag = (fy[:, None] ** 2 + fx[None, :] ** 2).sqrt()
        freq_mag = freq_mag / (freq_mag.max() + 1e-8)
        weight = 1.0 + (freq_weight - 1.0) * freq_mag
        noise_real = noise_real * weight
        noise_imag = noise_imag * weight

    # DDPM 순방향: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * noise
    a = sqrt_alphas_cumprod[t_idx]
    b = sqrt_one_minus_alphas_cumprod[t_idx]

    noisy_freq = torch.complex(
        a * X_freq.real + b * noise_real,
        a * X_freq.imag + b * noise_imag,
    )
    noise_spatial = torch.fft.irfft2(
        torch.complex(noise_real, noise_imag), s=(H, W), norm="ortho"
    )
    noisy_x = torch.fft.irfft2(noisy_freq, s=(H, W), norm="ortho")
    return noisy_x, noise_spatial


# ---------------------------------------------------------------------------
# FourierDiffusionScheduler
# ---------------------------------------------------------------------------

class FourierDiffusionScheduler:
    """
    DDPM / DDIM 스케줄러.

    Args:
        timesteps     : 전체 타임스텝 수
        schedule_type : 'cosine' 또는 'linear'
        beta_start    : 선형 스케줄 시작값
        beta_end      : 선형 스케줄 끝값
    """

    def __init__(
        self,
        timesteps: int = 1000,
        schedule_type: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        self.timesteps = timesteps

        if schedule_type == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register("betas", betas)
        self.register("alphas", alphas)
        self.register("alphas_cumprod", alphas_cumprod)
        self.register("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())
        self.register("log_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).log())
        self.register("sqrt_recip_alphas_cumprod", (1.0 / alphas_cumprod).sqrt())
        self.register("sqrt_recipm1_alphas_cumprod", (1.0 / alphas_cumprod - 1).sqrt())

        # DDPM posterior
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register("posterior_variance", posterior_variance)
        self.register(
            "posterior_log_variance_clipped",
            posterior_variance.clamp(min=1e-20).log(),
        )
        self.register(
            "posterior_mean_coef1",
            betas * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod),
        )
        self.register(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * alphas.sqrt() / (1.0 - alphas_cumprod),
        )

    def register(self, name: str, tensor: torch.Tensor):
        setattr(self, name, tensor)

    def to(self, device):
        for attr in [
            "betas", "alphas", "alphas_cumprod", "alphas_cumprod_prev",
            "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
            "log_one_minus_alphas_cumprod", "sqrt_recip_alphas_cumprod",
            "sqrt_recipm1_alphas_cumprod", "posterior_variance",
            "posterior_log_variance_clipped", "posterior_mean_coef1",
            "posterior_mean_coef2",
        ]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    # -----------------------------------------------------------------------
    # DDPM 샘플링 (단일 스텝)
    # -----------------------------------------------------------------------

    def p_mean_variance(self, model_output, x_t, t_idx: int):
        """DDPM posterior 평균 및 분산 계산."""
        # x_0 예측
        x0_pred = (
            self.sqrt_recip_alphas_cumprod[t_idx] * x_t
            - self.sqrt_recipm1_alphas_cumprod[t_idx] * model_output
        ).clamp(-1.0, 1.0)

        posterior_mean = (
            self.posterior_mean_coef1[t_idx] * x0_pred
            + self.posterior_mean_coef2[t_idx] * x_t
        )
        posterior_log_var = self.posterior_log_variance_clipped[t_idx]
        return posterior_mean, posterior_log_var, x0_pred

    def ddpm_step(self, model_output, x_t, t_idx: int, eta: float = 1.0):
        """DDPM 역방향 단일 스텝."""
        mean, log_var, x0_pred = self.p_mean_variance(model_output, x_t, t_idx)
        noise = torch.randn_like(x_t) if t_idx > 0 else torch.zeros_like(x_t)
        return mean + eta * (0.5 * log_var).exp() * noise, x0_pred

    # -----------------------------------------------------------------------
    # DDIM 샘플링 (단일 스텝, 결정론적)
    # -----------------------------------------------------------------------

    def ddim_step(
        self,
        model_output,
        x_t,
        t_idx: int,
        t_prev_idx: int,
        eta: float = 0.0,
    ):
        """
        DDIM 역방향 단일 스텝.
        eta=0 → 완전 결정론적, eta=1 → DDPM과 동일.
        """
        alpha_t = self.alphas_cumprod[t_idx]
        alpha_prev = self.alphas_cumprod[t_prev_idx] if t_prev_idx >= 0 else torch.tensor(1.0)

        # x_0 예측
        x0_pred = (x_t - (1 - alpha_t).sqrt() * model_output) / alpha_t.sqrt()
        x0_pred = x0_pred.clamp(-1.0, 1.0)

        # 방향 노이즈
        sigma = eta * ((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)).sqrt()
        dir_xt = (1 - alpha_prev - sigma ** 2).clamp(min=0).sqrt() * model_output
        noise = sigma * torch.randn_like(x_t) if eta > 0 else 0.0

        x_prev = alpha_prev.sqrt() * x0_pred + dir_xt + noise
        return x_prev, x0_pred

    # -----------------------------------------------------------------------
    # 완전 샘플링 루프
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        model,
        shape: tuple,
        device,
        sampler: str = "ddim",
        ddim_steps: int = 50,
        eta: float = 0.0,
        freq_weight: float = 1.0,
        cfg_scale: float = 1.0,
        callback=None,
    ) -> torch.Tensor:
        """
        완전 샘플링 루프.

        Args:
            model       : FourierDiffusionUNet
            shape       : (B, C, H, W)
            device      : torch device
            sampler     : 'ddpm' 또는 'ddim'
            ddim_steps  : DDIM 샘플링 스텝 수
            eta         : DDIM eta (0=결정론적)
            freq_weight : 초기 주파수 노이즈 가중치
            cfg_scale   : Classifier-Free Guidance 스케일 (미래 확장용)
            callback    : 진행 콜백 fn(step, total, x_t)
        Returns:
            생성된 이미지 텐서 [B, C, H, W], 값 범위 [-1, 1]
        """
        model.eval()
        B, C, H, W = shape
        scheduler = self.to(device)

        # 초기 주파수 가중 노이즈 생성
        x = torch.randn(shape, device=device)
        if freq_weight != 1.0:
            X_freq = torch.fft.rfft2(x, norm="ortho")
            fH, fW = X_freq.shape[-2], X_freq.shape[-1]
            fy = torch.fft.fftfreq(H, device=device)[:fH].abs()
            fx = torch.fft.rfftfreq(W, device=device)[:fW]
            freq_mag = (fy[:, None] ** 2 + fx[None, :] ** 2).sqrt()
            freq_mag = freq_mag / (freq_mag.max() + 1e-8)
            weight = 1.0 + (freq_weight - 1.0) * freq_mag
            X_freq_weighted = torch.complex(
                X_freq.real * weight, X_freq.imag * weight
            )
            x = torch.fft.irfft2(X_freq_weighted, s=(H, W), norm="ortho")

        if sampler == "ddim":
            # DDIM: 균일 간격 타임스텝 서브셋 (고→저 순서)
            step_ratio = max(1, self.timesteps // ddim_steps)
            # [T-1, T-1-r, T-1-2r, ...] 형태로 고→저 정렬
            ts = list(range(self.timesteps - 1, -1, -step_ratio))[:ddim_steps]
            # t_prev: 한 스텝 앞 인덱스, 마지막은 -1 (초기 상태)
            ts_prev = ts[1:] + [-1]
            total = len(ts)

            for i, (t_idx, t_prev_idx) in enumerate(zip(ts, ts_prev)):
                t_tensor = torch.full((B,), t_idx, device=device, dtype=torch.long)
                pred_noise = model(x, t_tensor)
                # NaN 방지: 모델 출력 클램핑
                pred_noise = pred_noise.nan_to_num(0.0).clamp(-10.0, 10.0)
                x_prev, _ = scheduler.ddim_step(
                    pred_noise, x, t_idx,
                    t_prev_idx=max(t_prev_idx, 0),
                    eta=eta,
                )
                x = x_prev
                if callback:
                    callback(i + 1, total, x)
        else:
            # DDPM: 전체 타임스텝
            total = self.timesteps
            for t_idx in reversed(range(self.timesteps)):
                t_tensor = torch.full((B,), t_idx, device=device, dtype=torch.long)
                pred_noise = model(x, t_tensor)
                pred_noise = pred_noise.nan_to_num(0.0).clamp(-10.0, 10.0)
                x, _ = scheduler.ddpm_step(pred_noise, x, t_idx)
                if callback:
                    callback(self.timesteps - t_idx, total, x)

        return x
