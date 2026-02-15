"""
FourierDiffusion Model Architecture
====================================
주파수(FFT) 도메인에서 동작하는 확산 모델 UNet.

핵심 아이디어:
 - rfft2로 특징맵을 주파수 도메인으로 변환
 - 주파수 계수를 처리하는 FourierResBlock + SpectralAttention
 - DDPM 스타일 타임스텝 임베딩
 - 스킵 연결: 인코더 각 ResBlock 출력 + 입력 투영 출력
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 타임스텝 임베딩
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


# ---------------------------------------------------------------------------
# 주파수 도메인 ResBlock
# ---------------------------------------------------------------------------

class FourierResBlock(nn.Module):
    """
    주파수 도메인 ResBlock.
    공간 도메인 입력 → rfft2 → 실수/허수 채널 처리 → irfft2 + 잔차.
    """

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(max(1, min(8, in_channels)), in_channels)
        self.norm2 = nn.GroupNorm(max(1, min(8, out_channels)), out_channels)

        self.freq_conv1 = nn.Conv2d(in_channels * 2, out_channels * 2, 3, padding=1)
        self.freq_conv2 = nn.Conv2d(out_channels * 2, out_channels * 2, 3, padding=1)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2),
        )
        self.dropout = nn.Dropout(dropout)

        self.skip_proj = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def _rfft(self, x: torch.Tensor):
        f = torch.fft.rfft2(x, norm="ortho")
        return torch.cat([f.real, f.imag], dim=1), x.shape[-2:]

    def _irfft(self, xf: torch.Tensor, shape):
        C = xf.shape[1] // 2
        return torch.fft.irfft2(torch.complex(xf[:, :C], xf[:, C:]), s=shape, norm="ortho")

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        skip = self.skip_proj(x)

        h = self.norm1(x)
        h_f, shape = self._rfft(h)
        h_f = F.silu(h_f)
        h_f = self.freq_conv1(h_f)
        h_f = h_f + self.time_proj(t_emb)[:, :, None, None]

        h = self._irfft(h_f, shape)
        h = self.norm2(h)
        h_f2, shape2 = self._rfft(h)
        h_f2 = F.silu(h_f2)
        h_f2 = self.dropout(h_f2)
        h_f2 = self.freq_conv2(h_f2)
        h = self._irfft(h_f2, shape2)

        return h + skip


# ---------------------------------------------------------------------------
# 스펙트럼 셀프-어텐션
# ---------------------------------------------------------------------------

class SpectralAttention(nn.Module):
    """rfft2 도메인 멀티헤드 셀프-어텐션."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        # num_heads가 channels를 나눌 수 있도록 조정
        while num_heads > 1 and channels % num_heads != 0:
            num_heads -= 1
        self.norm = nn.GroupNorm(max(1, min(8, channels)), channels)
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        f = torch.fft.rfft2(h, norm="ortho")
        fH, fW = f.shape[-2], f.shape[-1]
        N = fH * fW

        tokens = f.real.permute(0, 2, 3, 1).reshape(B, N, C)
        q, k, v = self.qkv(tokens).chunk(3, dim=-1)
        q, k, v = [t.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2) for t in (q, k, v)]

        attn = torch.softmax((q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5), dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out).reshape(B, fH, fW, C).permute(0, 3, 1, 2)

        return x + torch.fft.irfft2(torch.complex(out, f.imag), s=(H, W), norm="ortho")


# ---------------------------------------------------------------------------
# 다운/업샘플
# ---------------------------------------------------------------------------

class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


# ---------------------------------------------------------------------------
# FourierDiffusion UNet
# ---------------------------------------------------------------------------

class FourierDiffusionUNet(nn.Module):
    """
    주파수 도메인 확산 모델 UNet.

    스킵 연결 구조:
      - 인코더: input_proj 출력 + 각 레벨의 num_res_blocks 개 출력
        → 총 push = 1 + num_res_blocks * num_levels
      - 디코더: 각 레벨 num_res_blocks 개 pop + 마지막에 input_proj skip pop
        → 총 pop = num_res_blocks * num_levels + 1 (일치)

    Args:
        in_channels     : 입력/출력 채널 (3 = RGB)
        model_channels  : 기본 채널 수
        channel_mults   : 각 레벨 채널 배율 tuple
        num_res_blocks  : 레벨당 ResBlock 수
        attention_levels: 어텐션을 적용할 레벨 인덱스 set
        dropout         : 드롭아웃 비율
    """

    def __init__(
        self,
        in_channels: int = 3,
        model_channels: int = 64,
        channel_mults: tuple = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_levels: tuple = (2, 3),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.channel_mults = tuple(channel_mults)
        self.num_res_blocks = num_res_blocks
        self.num_levels = len(channel_mults)
        self.attention_levels = tuple(attention_levels)
        self.dropout = dropout
        attention_set = set(attention_levels)

        time_emb_dim = model_channels * 4
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.input_proj = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # ── 인코더 ──────────────────────────────────────────────────────────
        # enc_blocks[level] = ModuleList of (ResBlock, Attn) * num_res_blocks
        # enc_downs[level]  = Downsample (마지막 레벨 제외)
        self.enc_blocks = nn.ModuleList()
        self.enc_downs = nn.ModuleList()

        # skip_channels: 각 스킵의 채널 수 기록 (디코더 블록 크기 계산용)
        self._skip_ch = []
        ch = model_channels
        self._skip_ch.append(ch)  # input_proj 출력

        for level, mult in enumerate(channel_mults):
            out_ch = model_channels * mult
            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_blocks.append(FourierResBlock(ch, out_ch, time_emb_dim, dropout))
                level_blocks.append(SpectralAttention(out_ch) if level in attention_set else nn.Identity())
                ch = out_ch
                self._skip_ch.append(ch)
            self.enc_blocks.append(level_blocks)
            if level < len(channel_mults) - 1:
                self.enc_downs.append(Downsample(ch))

        # ── 병목 ────────────────────────────────────────────────────────────
        self.mid_block1 = FourierResBlock(ch, ch, time_emb_dim, dropout)
        self.mid_attn = SpectralAttention(ch)
        self.mid_block2 = FourierResBlock(ch, ch, time_emb_dim, dropout)

        # ── 디코더 ──────────────────────────────────────────────────────────
        # dec_blocks[level] = ModuleList of (ResBlock, Attn) * num_res_blocks
        # dec_ups[level]    = Upsample (첫 레벨 제외, reversed 기준)
        self.dec_blocks = nn.ModuleList()
        self.dec_ups = nn.ModuleList()

        # skip_ch 리스트를 역순으로 처리 (마지막 = input_proj skip)
        skip_ch_rev = list(reversed(self._skip_ch))  # 복사본

        for i, (level, mult) in enumerate(reversed(list(enumerate(channel_mults)))):
            out_ch = model_channels * mult
            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                skip_ch = skip_ch_rev.pop(0)
                level_blocks.append(FourierResBlock(ch + skip_ch, out_ch, time_emb_dim, dropout))
                level_blocks.append(SpectralAttention(out_ch) if level in attention_set else nn.Identity())
                ch = out_ch
            self.dec_blocks.append(level_blocks)
            if level > 0:
                self.dec_ups.append(Upsample(ch))

        # 마지막 스킵 (input_proj)을 사용하는 최종 블록
        final_skip_ch = skip_ch_rev.pop(0)  # input_proj의 채널
        self.final_block = FourierResBlock(ch + final_skip_ch, model_channels, time_emb_dim, dropout)

        self.output_proj = nn.Sequential(
            nn.GroupNorm(max(1, min(8, model_channels)), model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, in_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)

        # 입력 투영 + 스킵 저장
        h = self.input_proj(x)
        skips = [h]  # [input_proj_out, enc_level0_rb0, enc_level0_rb1, ...]

        # 인코더
        for level, level_blocks in enumerate(self.enc_blocks):
            for i in range(0, len(level_blocks), 2):
                h = level_blocks[i](h, t_emb)   # ResBlock
                h = level_blocks[i + 1](h)        # Attn or Identity
                skips.append(h)
            if level < len(self.enc_downs):
                h = self.enc_downs[level](h)

        # 병목
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # 디코더 (스킵을 역순으로 pop)
        up_idx = 0
        for level_blocks in self.dec_blocks:
            for i in range(0, len(level_blocks), 2):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = level_blocks[i](h, t_emb)
                h = level_blocks[i + 1](h)
            if up_idx < len(self.dec_ups):
                h = self.dec_ups[up_idx](h)
                up_idx += 1

        # 최종 블록 (input_proj 스킵)
        skip = skips.pop()
        h = torch.cat([h, skip], dim=1)
        h = self.final_block(h, t_emb)

        return self.output_proj(h)
