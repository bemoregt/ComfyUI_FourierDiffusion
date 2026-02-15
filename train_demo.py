"""
FourierDiffusion 학습 데모 스크립트
=====================================
ComfyUI 없이 독립 실행 가능한 학습 예제.

사용법:
    python train_demo.py --data_dir /path/to/images --output_dir ./checkpoints
    python train_demo.py --output_dir ./checkpoints  # 랜덤 데이터로 오버피팅 테스트
"""

import argparse
import os
import sys
import torch
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# 패키지 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from ComfyUI_FourierDiffusion.model import FourierDiffusionUNet
from ComfyUI_FourierDiffusion.diffusion import FourierDiffusionScheduler
from ComfyUI_FourierDiffusion.utils import save_checkpoint, load_checkpoint, train_step


# ---------------------------------------------------------------------------
# 데이터셋
# ---------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root: str, size: int = 256):
        self.paths = [
            p for p in Path(root).rglob("*")
            if p.suffix.lower() in self.EXTS
        ]
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # → [-1, 1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


class RandomDataset(Dataset):
    """학습 파이프라인 검증용 랜덤 데이터셋."""
    def __init__(self, size: int = 64, n: int = 100):
        self.data = torch.randn(n, 3, size, size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ---------------------------------------------------------------------------
# 메인 학습 루프
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None,
                        help="학습 이미지 폴더 (없으면 랜덤 데이터 사용)")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--model_channels", type=int, default=32)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--resume", type=str, default=None,
                        help="재개할 체크포인트 경로")
    parser.add_argument("--save_every", type=int, default=5,
                        help="N 에폭마다 체크포인트 저장")
    parser.add_argument("--loss_in_freq", action="store_true", default=True,
                        help="주파수 도메인 MSE 손실 사용")
    args = parser.parse_args()

    # 장치
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"장치: {device}")

    # 데이터셋
    if args.data_dir:
        dataset = ImageFolderDataset(args.data_dir, args.image_size)
        print(f"데이터셋: {len(dataset)}개 이미지")
    else:
        dataset = RandomDataset(args.image_size, n=200)
        print("데이터셋: 랜덤 (오버피팅 테스트)")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=(device.type == "cuda"))

    # 모델
    model = FourierDiffusionUNet(
        in_channels=3,
        model_channels=args.model_channels,
        channel_mults=(1, 2, 4),
        num_res_blocks=1,
        attention_levels=(1, 2),
        dropout=0.1,
    ).to(device)
    print(f"모델 파라미터: {sum(p.numel() for p in model.parameters()):,}")

    # 스케줄러
    scheduler = FourierDiffusionScheduler(
        timesteps=args.timesteps, schedule_type="cosine"
    ).to(device)

    # 옵티마이저
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # 재개
    start_epoch = 0
    if args.resume:
        start_epoch, _ = load_checkpoint(args.resume, model, optimizer, device)

    os.makedirs(args.output_dir, exist_ok=True)

    # 학습 루프
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        for batch_idx, batch in enumerate(loader):
            loss = train_step(
                model, scheduler, batch, optimizer, device,
                loss_in_freq=args.loss_in_freq,
            )
            total_loss += loss
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{args.epochs} "
                      f"Step {batch_idx}/{len(loader)} "
                      f"Loss: {loss:.4f}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} 완료 | 평균 손실: {avg_loss:.4f}")

        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, f"fourier_diffusion_epoch{epoch+1}.pt")
            save_checkpoint(model, ckpt_path, scheduler=scheduler,
                            optimizer=optimizer, epoch=epoch+1)

    # 최종 저장
    final_path = os.path.join(args.output_dir, "fourier_diffusion_final.pt")
    save_checkpoint(model, final_path, scheduler=scheduler,
                    optimizer=optimizer, epoch=args.epochs)
    print(f"\n학습 완료! 최종 체크포인트: {final_path}")


if __name__ == "__main__":
    main()
