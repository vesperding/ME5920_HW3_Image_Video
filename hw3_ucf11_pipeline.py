from __future__ import annotations

import argparse
import copy
import json
import random
import re
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    from torchvision.models import ResNet18_Weights, resnet18
    from torchvision.models.video import R3D_18_Weights, r3d_18
    HAS_TORCHVISION = True
except Exception:
    HAS_TORCHVISION = False


def resolve_project_root() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


PROJECT_ROOT = resolve_project_root()
DATA_ROOT = PROJECT_ROOT / "data" / "UCF11_updated_mpg"
SPLIT_ROOT = PROJECT_ROOT / "splits"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


# -----------------------------------------------------------------------------
# Reproducibility and basic utilities
# -----------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: object, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def normalize_tensor(x: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(x.device)
    std = IMAGENET_STD.to(x.device)
    return (x - mean) / std


def frames_to_tensor(frames: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(frames).float() / 255.0
    x = x.permute(0, 3, 1, 2).contiguous()  # (T, C, H, W)
    x = normalize_tensor(x)
    return x


def safe_resize_rgb(frame: np.ndarray, img_size: int) -> np.ndarray:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return frame


def read_frames_by_indices(video_path: str, indices: Sequence[int], img_size: int) -> np.ndarray:
    """
    Safe random-access reader.
    It is not the fastest possible implementation, but it is simple and robust.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return np.zeros((len(indices), img_size, img_size, 3), dtype=np.uint8)

    frames: List[np.ndarray] = []
    for idx in indices:
        idx = int(np.clip(idx, 0, total - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            frame = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        else:
            frame = safe_resize_rgb(frame, img_size)
        frames.append(frame)

    cap.release()
    return np.stack(frames, axis=0)


def parse_folder_index(folder_name: str) -> Optional[int]:
    match = re.search(r"_(\d+)$", folder_name)
    return int(match.group(1)) if match else None


def get_num_frames(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(total, 0)


def print_runtime_paths(data_root: Path, split_root: Path, output_root: Path) -> None:
    print("PROJECT_ROOT =", PROJECT_ROOT)
    print("DATA_ROOT    =", data_root.resolve())
    print("SPLIT_ROOT   =", split_root.resolve())
    print("OUTPUT_ROOT  =", output_root.resolve())


# -----------------------------------------------------------------------------
# Dataset scanning and split generation
# -----------------------------------------------------------------------------

def scan_dataset(data_root: Path) -> pd.DataFrame:
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_root}")

    class_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])
    label_map = {p.name: idx for idx, p in enumerate(class_dirs)}
    records: List[Dict[str, object]] = []

    for class_dir in class_dirs:
        group_dirs = sorted([p for p in class_dir.iterdir() if p.is_dir() and p.name.lower() != "annotation"])
        for group_dir in group_dirs:
            folder_idx = parse_folder_index(group_dir.name)
            if folder_idx is None:
                continue
            for video_path in sorted(group_dir.glob("*.mpg")):
                records.append(
                    {
                        "class_name": class_dir.name,
                        "label": label_map[class_dir.name],
                        "group_folder": group_dir.name,
                        "folder_idx": folder_idx,
                        "video_name": video_path.name,
                        "video_path": str(video_path.as_posix()),
                        "num_frames": get_num_frames(str(video_path.as_posix())),
                    }
                )

    if not records:
        raise RuntimeError(f"No .mpg files were found under {data_root}")

    return pd.DataFrame(records).sort_values(["class_name", "folder_idx", "video_name"]).reset_index(drop=True)


def assign_split(folder_idx: int) -> str:
    if 20 <= folder_idx <= 25:
        return "test"
    if 17 <= folder_idx <= 19:
        return "val"
    return "train"


def make_splits(
    data_root: Path = DATA_ROOT,
    split_root: Path = SPLIT_ROOT,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    df = scan_dataset(data_root)
    df["split"] = df["folder_idx"].apply(assign_split)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    assert test_df["folder_idx"].between(20, 25).all(), "Test split must use folders 20-25."

    label_map = {
        name: int(label)
        for name, label in df[["class_name", "label"]]
        .drop_duplicates()
        .sort_values("label")
        .values
    }

    ensure_dir(split_root)
    save_dataframe(df, split_root / "all_videos.csv")
    save_dataframe(train_df, split_root / "train.csv")
    save_dataframe(val_df, split_root / "val.csv")
    save_dataframe(test_df, split_root / "test.csv")
    save_json(label_map, split_root / "label_map.json")

    print(f"Total videos: {len(df)}")
    print(f"Train / Val / Test = {len(train_df)} / {len(val_df)} / {len(test_df)}")
    return train_df, val_df, test_df, label_map


def load_splits(split_root: Path = SPLIT_ROOT) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    required = ["train.csv", "val.csv", "test.csv", "label_map.json"]
    for name in required:
        if not (split_root / name).exists():
            raise FileNotFoundError(f"Missing split file: {split_root / name}")

    train_df = pd.read_csv(split_root / "train.csv")
    val_df = pd.read_csv(split_root / "val.csv")
    test_df = pd.read_csv(split_root / "test.csv")
    with open(split_root / "label_map.json", "r", encoding="utf-8") as f:
        label_map = json.load(f)
    return train_df, val_df, test_df, label_map


# -----------------------------------------------------------------------------
# Sampling helpers
# -----------------------------------------------------------------------------

def sample_sparse_indices(total_frames: int, num_samples: int, training: bool, jitter: int = 2) -> List[int]:
    if total_frames <= 0:
        return [0] * num_samples

    if total_frames < num_samples:
        indices = list(range(total_frames))
        indices.extend([total_frames - 1] * (num_samples - total_frames))
        return indices

    bins = np.linspace(0, total_frames, num_samples + 1).astype(int)
    out: List[int] = []
    for i in range(num_samples):
        lo = bins[i]
        hi = max(bins[i + 1] - 1, lo)
        if training:
            pos = random.randint(lo, hi)
            pos += random.randint(-jitter, jitter)
            pos = int(np.clip(pos, lo, hi))
        else:
            pos = (lo + hi) // 2
        out.append(pos)
    out.sort()
    return out


def sample_clip_indices(
    total_frames: int,
    clip_len: int,
    sample_rate: int,
    training: bool,
    temporal_crop_id: int = 0,
    num_temporal_crops: int = 1,
) -> List[int]:
    if total_frames <= 0:
        return [0] * clip_len

    span = max(1, clip_len * sample_rate)
    if total_frames <= span:
        base = np.linspace(0, total_frames - 1, clip_len).astype(int).tolist()
        while len(base) < clip_len:
            base.append(total_frames - 1)
        return base[:clip_len]

    max_start = total_frames - span
    if training:
        start = random.randint(0, max_start)
    else:
        if num_temporal_crops <= 1:
            start = max_start // 2
        else:
            starts = np.linspace(0, max_start, num_temporal_crops).astype(int).tolist()
            start = starts[min(temporal_crop_id, len(starts) - 1)]

    return [start + i * sample_rate for i in range(clip_len)]


def sample_task4_sequence_indices(
    total_frames: int,
    seq_len: int,
    training: bool,
    base_stride: int = 2,
    temporal_crop_id: int = 0,
    num_temporal_crops: int = 1,
    phase_id: int = 0,
    num_phases: int = 1,
) -> List[int]:
    """
    Sparse sequence sampler for Task 4.

    Training:
    - random start with small local jitter

    Evaluation:
    - deterministic temporal crops across the video
    - optional deterministic phase shifts to reduce validation/test jitter
    """
    if total_frames <= 0:
        return [0] * seq_len

    stride = base_stride if total_frames >= seq_len * base_stride else 1
    span = (seq_len - 1) * stride + 1

    if total_frames <= span:
        base = np.linspace(0, total_frames - 1, seq_len).astype(int).tolist()
        while len(base) < seq_len:
            base.append(total_frames - 1)
        return base[:seq_len]

    max_start = total_frames - span

    if training:
        start = random.randint(0, max_start)
    else:
        if num_temporal_crops <= 1:
            start = max_start // 2
        else:
            starts = np.linspace(0, max_start, num_temporal_crops).astype(int).tolist()
            start = starts[min(temporal_crop_id, len(starts) - 1)]

        if num_phases > 1:
            local_radius = max(1, stride)
            phase_offsets = np.linspace(-local_radius, local_radius, num_phases)
            shift = int(round(float(phase_offsets[min(phase_id, num_phases - 1)])))
            start = int(np.clip(start + shift, 0, max_start))

    indices = [start + i * stride for i in range(seq_len)]

    if training:
        jittered: List[int] = []
        prev = -1
        for idx in indices:
            j = int(np.clip(idx + random.randint(-1, 1), 0, total_frames - 1))
            if j < prev:
                j = prev
            jittered.append(j)
            prev = j
        indices = jittered

    return indices


# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------

def safe_get_num_frames(row_or_series) -> int:
    if isinstance(row_or_series, pd.Series):
        if "num_frames" in row_or_series.index and pd.notna(row_or_series["num_frames"]):
            return int(row_or_series["num_frames"])
        video_path = str(row_or_series["video_path"])
    else:
        val = getattr(row_or_series, "num_frames", None)
        if val is not None and not pd.isna(val):
            return int(val)
        video_path = str(getattr(row_or_series, "video_path"))
    return get_num_frames(video_path)


class FrameDataset2D(Dataset):
    def __init__(self, df: pd.DataFrame, img_size: int = 112, training: bool = True):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.training = training

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        total = safe_get_num_frames(row)
        if total <= 0:
            indices = [0]
        elif self.training:
            indices = [random.randint(0, total - 1)]
        else:
            indices = [total // 2]
        frames = read_frames_by_indices(str(row["video_path"]), indices, self.img_size)
        x = frames_to_tensor(frames)[0]
        y = int(row["label"])
        return x, y


class ClipDataset3D(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        clip_len: int = 16,
        sample_rate: int = 2,
        img_size: int = 112,
        training: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.clip_len = clip_len
        self.sample_rate = sample_rate
        self.img_size = img_size
        self.training = training

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        total = safe_get_num_frames(row)
        indices = sample_clip_indices(
            total_frames=total,
            clip_len=self.clip_len,
            sample_rate=self.sample_rate,
            training=self.training,
        )
        frames = read_frames_by_indices(str(row["video_path"]), indices, self.img_size)
        x = frames_to_tensor(frames).permute(1, 0, 2, 3).contiguous()  # (C, T, H, W)
        y = int(row["label"])
        return x, y


class SequenceDatasetLSTM(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = 12,
        img_size: int = 112,
        training: bool = True,
        base_stride: int = 2,
    ):
        self.df = df.reset_index(drop=True)
        self.seq_len = seq_len
        self.img_size = img_size
        self.training = training
        self.base_stride = base_stride

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        total = safe_get_num_frames(row)
        indices = sample_task4_sequence_indices(
            total_frames=total,
            seq_len=self.seq_len,
            training=self.training,
            base_stride=self.base_stride,
            temporal_crop_id=0,
            num_temporal_crops=1,
        )
        frames = read_frames_by_indices(str(row["video_path"]), indices, self.img_size)
        x = frames_to_tensor(frames)  # (T, C, H, W)
        y = int(row["label"])
        return x, y


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

class Small2DCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Small3DCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallCNNEncoder(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = self.attn(x).squeeze(-1)
        weight = torch.softmax(score, dim=1)
        return torch.sum(x * weight.unsqueeze(-1), dim=1)


class ResNetBiLSTMAttention(nn.Module):
    """
    Task 4 model.
    The encoder supports three tuning modes:
    - freeze: use the pretrained encoder as a fixed feature extractor
    - partial: fine-tune only ResNet layer4
    - full: fine-tune the full encoder
    """

    def __init__(
        self,
        num_classes: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.4,
        bidirectional: bool = True,
        encoder_tune_mode: str = "partial",
    ):
        super().__init__()
        assert encoder_tune_mode in {"freeze", "partial", "full"}
        self.encoder_tune_mode = encoder_tune_mode
        self.using_torchvision = False
        self.feat_dim = 256

        if HAS_TORCHVISION:
            try:
                backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
                backbone.fc = nn.Identity()
                self.backbone = backbone
                self.feat_dim = 512
                self.using_torchvision = True
            except Exception:
                self.encoder = SmallCNNEncoder(out_dim=self.feat_dim)
        else:
            self.encoder = SmallCNNEncoder(out_dim=self.feat_dim)

        if self.using_torchvision:
            if self.encoder_tune_mode == "freeze":
                for p in self.backbone.parameters():
                    p.requires_grad = False
            elif self.encoder_tune_mode == "partial":
                for p in self.backbone.parameters():
                    p.requires_grad = False
                for p in self.backbone.layer4.parameters():
                    p.requires_grad = True
            else:
                for p in self.backbone.parameters():
                    p.requires_grad = True
        else:
            for p in self.encoder.parameters():
                p.requires_grad = self.encoder_tune_mode != "freeze"

        self.feat_norm = nn.Sequential(nn.LayerNorm(self.feat_dim), nn.Dropout(0.2))
        self.lstm = nn.LSTM(
            input_size=self.feat_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.temporal_pool = TemporalAttention(out_dim)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(out_dim, num_classes))

    def _forward_backbone_full(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def _forward_backbone_partial(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def train(self, mode: bool = True):
        super().train(mode)
        if self.using_torchvision:
            if self.encoder_tune_mode == "freeze":
                self.backbone.eval()
            elif self.encoder_tune_mode == "partial":
                self.backbone.conv1.eval()
                self.backbone.bn1.eval()
                self.backbone.layer1.eval()
                self.backbone.layer2.eval()
                self.backbone.layer3.eval()
                self.backbone.layer4.train(mode)
        else:
            if self.encoder_tune_mode == "freeze":
                self.encoder.eval()
        return self

    def encode_frames(self, x: torch.Tensor) -> torch.Tensor:
        if self.using_torchvision:
            if self.encoder_tune_mode == "freeze":
                self.backbone.eval()
                with torch.no_grad():
                    feats = self._forward_backbone_full(x)
            elif self.encoder_tune_mode == "partial":
                feats = self._forward_backbone_partial(x)
            else:
                feats = self._forward_backbone_full(x)
        else:
            if self.encoder_tune_mode == "freeze":
                self.encoder.eval()
                with torch.no_grad():
                    feats = self.encoder(x)
            else:
                feats = self.encoder(x)

        if feats.ndim == 4:
            feats = feats.flatten(1)
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        feats = self.encode_frames(x)
        feats = self.feat_norm(feats)
        feats = feats.reshape(b, t, -1)
        seq, _ = self.lstm(feats)
        pooled = self.temporal_pool(seq)
        return self.classifier(pooled)


def build_task2_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    if HAS_TORCHVISION:
        try:
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            model = resnet18(weights=weights)
            in_dim = model.fc.in_features
            model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_dim, num_classes))
            return model
        except Exception:
            pass
    return Small2DCNN(num_classes)


def build_task3_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    if HAS_TORCHVISION:
        try:
            weights = R3D_18_Weights.DEFAULT if pretrained else None
            model = r3d_18(weights=weights, progress=True)
            in_dim = model.fc.in_features
            model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_dim, num_classes))
            return model
        except Exception:
            pass
    return Small3DCNN(num_classes)


# -----------------------------------------------------------------------------
# Training utilities
# -----------------------------------------------------------------------------

@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 16
    num_workers: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    early_stop_patience: int = 6
    use_weighted_sampler: bool = True
    use_amp: bool = True
    max_grad_norm: float = 5.0


class EarlyStopper:
    def __init__(self, patience: int = 6, mode: str = "max", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            return False
        if self.mode == "max":
            improved = value > self.best + self.min_delta
        else:
            improved = value < self.best - self.min_delta
        if improved:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def build_weighted_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    class_counts = df["label"].value_counts().sort_index()
    weights = {int(cls): 1.0 / float(cnt) for cls, cnt in class_counts.items()}
    sample_weights = df["label"].map(weights).astype(float).values
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def make_class_weights(df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    counts = df["label"].value_counts().sort_index()
    inv = 1.0 / counts.astype(float)
    weights = inv / inv.mean()
    return torch.as_tensor(weights.values, dtype=torch.float32, device=device)


def create_loader(
    dataset: Dataset,
    batch_size: int,
    training: bool,
    num_workers: int,
    weighted_sampler: Optional[WeightedRandomSampler] = None,
) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if training and weighted_sampler is not None:
        return DataLoader(dataset, sampler=weighted_sampler, shuffle=False, **kwargs)
    return DataLoader(dataset, shuffle=training, **kwargs)


def make_grad_scaler(use_amp: bool, device: torch.device):
    enabled = bool(use_amp and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(enabled: bool, device: torch.device):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast(device_type=device.type, enabled=True)
        except TypeError:
            return torch.amp.autocast(enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler=None,
    max_grad_norm: float = 5.0,
) -> Tuple[float, float, float]:
    model.train(True)
    losses: List[float] = []
    y_true: List[int] = []
    y_pred: List[int] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        use_amp = scaler is not None and device.type == "cuda"
        with autocast_context(use_amp, device):
            logits = model(x)
            loss = criterion(logits, y)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        pred = torch.argmax(logits, dim=1)
        losses.append(float(loss.item()))
        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(pred.detach().cpu().tolist())

    acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) if y_true else 0.0
    return float(np.mean(losses) if losses else 0.0), float(acc), float(macro_f1)


@torch.inference_mode()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: Optional[nn.Module],
    device: torch.device,
) -> Tuple[float, float, float, List[int], List[int]]:
    model.eval()
    losses: List[float] = []
    y_true: List[int] = []
    y_pred: List[int] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        if criterion is not None:
            losses.append(float(criterion(logits, y).item()))
        pred = torch.argmax(logits, dim=1)
        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(pred.detach().cpu().tolist())

    acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) if y_true else 0.0
    return float(np.mean(losses) if losses else 0.0), float(acc), float(macro_f1), y_true, y_pred


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_df: pd.DataFrame,
    device: torch.device,
    cfg: TrainConfig,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    criterion = nn.CrossEntropyLoss(
        weight=make_class_weights(train_df, device=device),
        label_smoothing=cfg.label_smoothing,
    )
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    scaler = make_grad_scaler(cfg.use_amp, device)
    stopper = EarlyStopper(patience=cfg.early_stop_patience, mode="max")

    history = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "lr": [],
    }

    best_state = copy.deepcopy(model.state_dict())
    best_f1 = -1.0

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            max_grad_norm=cfg.max_grad_norm,
        )
        va_loss, va_acc, va_f1, _, _ = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step(va_f1)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["train_f1"].append(tr_f1)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["val_f1"].append(va_f1)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        if va_f1 > best_f1:
            best_f1 = va_f1
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} tr_f1={tr_f1:.4f} | "
            f"va_loss={va_loss:.4f} va_acc={va_acc:.4f} va_f1={va_f1:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | {time.time() - t0:.1f}s"
        )

        if stopper.step(va_f1):
            print("Early stopping triggered.")
            break

    model.load_state_dict(best_state)
    return model, history


def build_task4_optimizer(
    model: nn.Module,
    head_lr: float = 5e-4,
    encoder_lr: float = 1e-4,
    weight_decay: float = 1e-4,
) -> optim.Optimizer:
    param_groups = []
    used = set()

    def add_group(params: Iterable[torch.nn.Parameter], lr: float) -> None:
        params = [p for p in params if p.requires_grad and id(p) not in used]
        if not params:
            return
        param_groups.append({"params": params, "lr": lr})
        for p in params:
            used.add(id(p))

    if hasattr(model, "backbone"):
        add_group(model.backbone.parameters(), encoder_lr)
    elif hasattr(model, "encoder"):
        add_group(model.encoder.parameters(), encoder_lr)

    add_group(model.feat_norm.parameters(), head_lr)
    add_group(model.lstm.parameters(), head_lr)
    add_group(model.temporal_pool.parameters(), head_lr)
    add_group(model.classifier.parameters(), head_lr)
    return optim.AdamW(param_groups, weight_decay=weight_decay)


def fit_task4_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: TrainConfig,
    device: torch.device,
    seq_len: int,
    img_size: int,
    base_stride: int,
    selection_seq_candidates: Sequence[int] = (1, 3),
    selection_num_repeats: int = 3,
    head_lr: float = 5e-4,
    encoder_lr: float = 1e-4,
) -> Tuple[nn.Module, Dict[str, List[float]], int]:
    criterion = nn.CrossEntropyLoss(
        weight=make_class_weights(train_df, device=device),
        label_smoothing=cfg.label_smoothing,
    )
    optimizer = build_task4_optimizer(
        model,
        head_lr=head_lr,
        encoder_lr=encoder_lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    scaler = make_grad_scaler(cfg.use_amp, device)
    stopper = EarlyStopper(patience=cfg.early_stop_patience, mode="max")

    history = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_select_acc": [],
        "val_select_f1": [],
        "val_select_num_sequences": [],
        "encoder_lr": [],
        "head_lr": [],
    }

    best_state = copy.deepcopy(model.state_dict())
    best_select_f1 = -1.0
    best_select_acc = -1.0
    best_num_sequences = int(selection_seq_candidates[0])

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            max_grad_norm=cfg.max_grad_norm,
        )
        va_loss, va_acc, va_f1, _, _ = evaluate_epoch(model, val_loader, criterion, device)

        val_select_df, _ = run_task4_sequence_sweep(
            model=model,
            df=val_df,
            device=device,
            seq_len=seq_len,
            img_size=img_size,
            base_stride=base_stride,
            seq_candidates=selection_seq_candidates,
            num_repeats=selection_num_repeats,
        )
        best_row = val_select_df.sort_values(
            ["macro_f1", "accuracy", "num_sequences"],
            ascending=[False, False, True],
        ).iloc[0]
        val_select_acc = float(best_row["accuracy"])
        val_select_f1 = float(best_row["macro_f1"])
        val_select_num_sequences = int(best_row["num_sequences"])

        scheduler.step(val_select_f1)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["train_f1"].append(tr_f1)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["val_f1"].append(va_f1)
        history["val_select_acc"].append(val_select_acc)
        history["val_select_f1"].append(val_select_f1)
        history["val_select_num_sequences"].append(val_select_num_sequences)
        history["encoder_lr"].append(float(optimizer.param_groups[0]["lr"]))
        history["head_lr"].append(float(optimizer.param_groups[-1]["lr"]))

        improved = (
            (val_select_f1 > best_select_f1 + 1e-4)
            or (
                abs(val_select_f1 - best_select_f1) <= 1e-4
                and val_select_acc > best_select_acc + 1e-4
            )
        )
        if improved:
            best_select_f1 = val_select_f1
            best_select_acc = val_select_acc
            best_num_sequences = val_select_num_sequences
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} tr_f1={tr_f1:.4f} | "
            f"va_loss={va_loss:.4f} va_acc={va_acc:.4f} va_f1={va_f1:.4f} | "
            f"val_select_f1={val_select_f1:.4f} val_select_acc={val_select_acc:.4f} "
            f"val_select_seq={val_select_num_sequences} | "
            f"enc_lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"head_lr={optimizer.param_groups[-1]['lr']:.2e} | {time.time() - t0:.1f}s"
        )

        if stopper.step(val_select_f1):
            print("Early stopping triggered by stable multi-sequence validation.")
            break

    model.load_state_dict(best_state)
    return model, history, int(best_num_sequences)


# -----------------------------------------------------------------------------
# Reporting helpers
# -----------------------------------------------------------------------------

def classification_report_dataframe(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    idx_to_class: Dict[int, str],
) -> pd.DataFrame:
    report = classification_report(
        y_true,
        y_pred,
        target_names=[idx_to_class[i] for i in range(len(idx_to_class))],
        zero_division=0,
        output_dict=True,
    )
    return pd.DataFrame(report).T.reset_index().rename(columns={"index": "class_name"})


def plot_history(history: Dict[str, List[float]], title_prefix: str, path: Path) -> None:
    ensure_dir(path.parent)
    fig = plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    if "train_loss" in history:
        plt.plot(history["train_loss"], label="train_loss")
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="val_loss")
    plt.title(f"{title_prefix} Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 3, 2)
    if "train_acc" in history:
        plt.plot(history["train_acc"], label="train_acc")
    if "val_acc" in history:
        plt.plot(history["val_acc"], label="val_acc")
    plt.title(f"{title_prefix} Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 3, 3)
    if "train_f1" in history:
        plt.plot(history["train_f1"], label="train_macro_f1")
    if "val_f1" in history:
        plt.plot(history["val_f1"], label="val_macro_f1")
    plt.title(f"{title_prefix} Macro-F1")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    idx_to_class: Dict[int, str],
    title: str,
    path: Path,
) -> None:
    ensure_dir(path.parent)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(idx_to_class))))
    names = [idx_to_class[i] for i in range(len(idx_to_class))]

    fig = plt.figure(figsize=(9, 7))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    plt.xticks(range(len(names)), names, rotation=45, ha="right")
    plt.yticks(range(len(names)), names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_results_summary(results_df: pd.DataFrame, path: Path) -> None:
    if len(results_df) == 0:
        return
    ensure_dir(path.parent)
    fig = plt.figure(figsize=(10, 4))
    x = np.arange(len(results_df))
    plt.bar(x, results_df["macro_f1"].values)
    plt.xticks(x, results_df["model"].tolist(), rotation=30, ha="right")
    plt.ylabel("Macro-F1")
    plt.title("Model Comparison")
    plt.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_prediction_table(
    df_eval: pd.DataFrame,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    idx_to_class: Dict[int, str],
) -> pd.DataFrame:
    out = df_eval.copy().reset_index(drop=True)
    out["true_label"] = list(y_true)
    out["pred_label"] = list(y_pred)
    out["true_name"] = out["true_label"].map(idx_to_class)
    out["pred_name"] = out["pred_label"].map(idx_to_class)
    out["correct"] = out["true_label"] == out["pred_label"]
    return out


# -----------------------------------------------------------------------------
# Task-specific evaluation
# -----------------------------------------------------------------------------

@torch.inference_mode()
def evaluate_task2_video_voting(
    model: nn.Module,
    df: pd.DataFrame,
    device: torch.device,
    img_size: int = 112,
    n_frames: int = 3,
    mode: str = "random",
    num_repeats: int = 3,
    eval_seed: int = 42,
) -> Dict[str, object]:
    assert mode in {"random", "segment"}
    model.eval()
    t0 = time.time()

    repeated_preds: List[List[int]] = []
    y_true: Optional[List[int]] = None

    for repeat_id in range(num_repeats if mode == "random" else 1):
        random.seed(eval_seed + repeat_id)
        preds_this_round: List[int] = []
        y_true_this_round: List[int] = []

        for _, row in df.iterrows():
            total = safe_get_num_frames(row)
            if total <= 0:
                indices = [0] * n_frames
            elif mode == "random":
                indices = sorted(random.randint(0, total - 1) for _ in range(n_frames))
            else:
                indices = sample_sparse_indices(total, n_frames, training=False, jitter=0)

            frames = read_frames_by_indices(str(row["video_path"]), indices, img_size)
            batch = frames_to_tensor(frames).to(device)  # (T, C, H, W)
            logits = model(batch)
            avg_logits = logits.mean(dim=0)
            pred = int(torch.argmax(avg_logits).item())

            preds_this_round.append(pred)
            y_true_this_round.append(int(row["label"]))

        repeated_preds.append(preds_this_round)
        y_true = y_true_this_round

    pred_matrix = np.asarray(repeated_preds)
    if pred_matrix.shape[0] == 1:
        final_pred = pred_matrix[0].tolist()
    else:
        final_pred = []
        for col in range(pred_matrix.shape[1]):
            values, counts = np.unique(pred_matrix[:, col], return_counts=True)
            final_pred.append(int(values[np.argmax(counts)]))

    assert y_true is not None
    elapsed = time.time() - t0
    return {
        "accuracy": float(accuracy_score(y_true, final_pred)),
        "macro_f1": float(f1_score(y_true, final_pred, average="macro", zero_division=0)),
        "avg_sec_per_video": float(elapsed / max(len(df), 1)),
        "y_true": y_true,
        "y_pred": final_pred,
    }


def run_task2_frame_sweep(
    model: nn.Module,
    df: pd.DataFrame,
    device: torch.device,
    img_size: int,
    n_candidates: Sequence[int],
    mode: str,
    num_repeats: int,
    eval_seed: int,
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, object]]]:
    results = []
    outputs_by_n = {}
    for n_frames in n_candidates:
        out = evaluate_task2_video_voting(
            model=model,
            df=df,
            device=device,
            img_size=img_size,
            n_frames=int(n_frames),
            mode=mode,
            num_repeats=num_repeats,
            eval_seed=eval_seed,
        )
        outputs_by_n[int(n_frames)] = out
        results.append(
            {
                "n_frames": int(n_frames),
                "accuracy": float(out["accuracy"]),
                "macro_f1": float(out["macro_f1"]),
                "avg_sec_per_video": float(out["avg_sec_per_video"]),
            }
        )
    return pd.DataFrame(results), outputs_by_n


@torch.inference_mode()
def evaluate_task3_multiclip(
    model: nn.Module,
    df: pd.DataFrame,
    device: torch.device,
    clip_len: int = 16,
    sample_rate: int = 2,
    img_size: int = 112,
    num_temporal_crops: int = 1,
) -> Dict[str, object]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    t0 = time.time()

    for _, row in df.iterrows():
        total = safe_get_num_frames(row)
        clip_logits = []
        for crop_id in range(num_temporal_crops):
            indices = sample_clip_indices(
                total_frames=total,
                clip_len=clip_len,
                sample_rate=sample_rate,
                training=False,
                temporal_crop_id=crop_id,
                num_temporal_crops=num_temporal_crops,
            )
            frames = read_frames_by_indices(str(row["video_path"]), indices, img_size)
            x = frames_to_tensor(frames).permute(1, 0, 2, 3).unsqueeze(0).to(device)
            clip_logits.append(model(x).squeeze(0))

        avg_logits = torch.stack(clip_logits, dim=0).mean(dim=0)
        pred = int(torch.argmax(avg_logits).item())
        y_true.append(int(row["label"]))
        y_pred.append(pred)

    elapsed = time.time() - t0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "avg_sec_per_video": float(elapsed / max(len(df), 1)),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def run_task3_crop_sweep(
    model: nn.Module,
    df: pd.DataFrame,
    device: torch.device,
    clip_len: int,
    sample_rate: int,
    img_size: int,
    crop_candidates: Sequence[int],
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, object]]]:
    results = []
    outputs_by_crop = {}
    for n_crops in crop_candidates:
        out = evaluate_task3_multiclip(
            model=model,
            df=df,
            device=device,
            clip_len=clip_len,
            sample_rate=sample_rate,
            img_size=img_size,
            num_temporal_crops=int(n_crops),
        )
        outputs_by_crop[int(n_crops)] = out
        results.append(
            {
                "num_temporal_crops": int(n_crops),
                "accuracy": float(out["accuracy"]),
                "macro_f1": float(out["macro_f1"]),
                "avg_sec_per_video": float(out["avg_sec_per_video"]),
            }
        )
    return pd.DataFrame(results), outputs_by_crop


@torch.inference_mode()
def evaluate_task4_multiseq(
    model: nn.Module,
    df: pd.DataFrame,
    device: torch.device,
    seq_len: int = 12,
    img_size: int = 112,
    base_stride: int = 2,
    num_sequences: int = 1,
    num_repeats: int = 3,
) -> Dict[str, object]:
    """
    Stable Task 4 evaluation.

    For each video, the model averages logits over:
    - multiple deterministic temporal crops (num_sequences)
    - multiple deterministic local phase shifts (num_repeats)

    This makes validation selection much more stable than using a single crop.
    """
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    t0 = time.time()

    for row in df.itertuples(index=False):
        total = safe_get_num_frames(row)
        seq_logits = []
        for repeat_id in range(num_repeats):
            for seq_id in range(num_sequences):
                indices = sample_task4_sequence_indices(
                    total_frames=total,
                    seq_len=seq_len,
                    training=False,
                    base_stride=base_stride,
                    temporal_crop_id=seq_id,
                    num_temporal_crops=num_sequences,
                    phase_id=repeat_id,
                    num_phases=num_repeats,
                )
                frames = read_frames_by_indices(str(row.video_path), indices, img_size)
                x = frames_to_tensor(frames).unsqueeze(0).to(device)
                seq_logits.append(model(x).squeeze(0))

        avg_logits = torch.stack(seq_logits, dim=0).mean(dim=0)
        pred = int(torch.argmax(avg_logits).item())
        y_true.append(int(row.label))
        y_pred.append(pred)

    elapsed = time.time() - t0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "avg_sec_per_video": float(elapsed / max(len(df), 1)),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def run_task4_sequence_sweep(
    model: nn.Module,
    df: pd.DataFrame,
    device: torch.device,
    seq_len: int,
    img_size: int,
    base_stride: int,
    seq_candidates: Sequence[int],
    num_repeats: int = 3,
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, object]]]:
    results = []
    outputs_by_seq = {}
    for n_seq in seq_candidates:
        out = evaluate_task4_multiseq(
            model=model,
            df=df,
            device=device,
            seq_len=seq_len,
            img_size=img_size,
            base_stride=base_stride,
            num_sequences=int(n_seq),
            num_repeats=int(num_repeats),
        )
        outputs_by_seq[int(n_seq)] = out
        results.append(
            {
                "num_sequences": int(n_seq),
                "num_repeats": int(num_repeats),
                "accuracy": float(out["accuracy"]),
                "macro_f1": float(out["macro_f1"]),
                "avg_sec_per_video": float(out["avg_sec_per_video"]),
            }
        )
    return pd.DataFrame(results), outputs_by_seq


# -----------------------------------------------------------------------------
# Task runners
# -----------------------------------------------------------------------------

def run_task2(split_root: Path, output_root: Path, device: torch.device, img_size: int = 112) -> Dict[str, object]:
    train_df, val_df, test_df, label_map = load_splits(split_root)
    idx_to_class = {v: k for k, v in label_map.items()}
    task_dir = ensure_dir(output_root / "task2")

    cfg = TrainConfig(
        epochs=20,
        batch_size=32,
        num_workers=2,
        lr=1e-3,
        weight_decay=1e-4,
        label_smoothing=0.05,
        early_stop_patience=6,
        use_weighted_sampler=True,
        use_amp=True,
    )

    train_ds = FrameDataset2D(train_df, img_size=img_size, training=True)
    val_ds = FrameDataset2D(val_df, img_size=img_size, training=False)
    sampler = build_weighted_sampler(train_df) if cfg.use_weighted_sampler else None
    train_loader = create_loader(train_ds, cfg.batch_size, True, cfg.num_workers, sampler)
    val_loader = create_loader(val_ds, cfg.batch_size, False, cfg.num_workers)

    model = build_task2_model(num_classes=len(label_map), pretrained=True).to(device)
    model, history = fit_model(model, train_loader, val_loader, train_df, device, cfg)
    torch.save(model.state_dict(), task_dir / "task2_best.pt")
    plot_history(history, "Task 2", task_dir / "task2_history.png")
    save_json(history, task_dir / "task2_history.json")

    n_candidates = [1, 3, 5]
    val_df_res, val_out_by_n = run_task2_frame_sweep(
        model, val_df, device, img_size, n_candidates, mode="random", num_repeats=3, eval_seed=42
    )
    test_df_res, test_out_by_n = run_task2_frame_sweep(
        model, test_df, device, img_size, n_candidates, mode="random", num_repeats=3, eval_seed=42
    )
    save_dataframe(val_df_res, task_dir / "task2_val_frame_compare.csv")
    save_dataframe(test_df_res, task_dir / "task2_test_frame_compare.csv")

    best_row = val_df_res.sort_values(["macro_f1", "accuracy"], ascending=False).iloc[0]
    best_n = int(best_row["n_frames"])
    best_out = test_out_by_n[best_n]

    segment_out = evaluate_task2_video_voting(
        model,
        test_df,
        device,
        img_size=img_size,
        n_frames=5,
        mode="segment",
        num_repeats=1,
        eval_seed=42,
    )
    random_vs_segment = pd.DataFrame(
        [
            {
                "mode": "random",
                "n_frames": 5,
                "accuracy": test_out_by_n[5]["accuracy"],
                "macro_f1": test_out_by_n[5]["macro_f1"],
                "avg_sec_per_video": test_out_by_n[5]["avg_sec_per_video"],
            },
            {
                "mode": "segment",
                "n_frames": 5,
                "accuracy": segment_out["accuracy"],
                "macro_f1": segment_out["macro_f1"],
                "avg_sec_per_video": segment_out["avg_sec_per_video"],
            },
        ]
    )
    save_dataframe(random_vs_segment, task_dir / "task2_random_vs_segment.csv")

    report_df = classification_report_dataframe(best_out["y_true"], best_out["y_pred"], idx_to_class)
    pred_df = save_prediction_table(test_df, best_out["y_true"], best_out["y_pred"], idx_to_class)
    save_dataframe(report_df, task_dir / "task2_classification_report.csv")
    save_dataframe(pred_df, task_dir / "task2_test_predictions.csv")
    plot_confusion_matrix(
        best_out["y_true"],
        best_out["y_pred"],
        idx_to_class,
        f"Task 2 Confusion Matrix ({best_n} frames)",
        task_dir / "task2_confusion.png",
    )

    summary = {
        "task": "Task2",
        "model": f"Task2_2DCNN_random_{best_n}frames",
        "accuracy": float(best_out["accuracy"]),
        "macro_f1": float(best_out["macro_f1"]),
        "avg_sec_per_video": float(best_out["avg_sec_per_video"]),
        "best_n_frames_by_val": best_n,
    }
    save_json(summary, task_dir / "task2_summary.json")
    print(summary)
    return summary


def run_task3(
    split_root: Path,
    output_root: Path,
    device: torch.device,
    img_size: int = 112,
    clip_len: int = 16,
    sample_rate: int = 2,
) -> Dict[str, object]:
    train_df, val_df, test_df, label_map = load_splits(split_root)
    idx_to_class = {v: k for k, v in label_map.items()}
    task_dir = ensure_dir(output_root / "task3")

    cfg = TrainConfig(
        epochs=20,
        batch_size=8,
        num_workers=2,
        lr=1e-4,
        weight_decay=1e-4,
        label_smoothing=0.05,
        early_stop_patience=6,
        use_weighted_sampler=True,
        use_amp=True,
    )

    train_ds = ClipDataset3D(train_df, clip_len=clip_len, sample_rate=sample_rate, img_size=img_size, training=True)
    val_ds = ClipDataset3D(val_df, clip_len=clip_len, sample_rate=sample_rate, img_size=img_size, training=False)
    sampler = build_weighted_sampler(train_df) if cfg.use_weighted_sampler else None
    train_loader = create_loader(train_ds, cfg.batch_size, True, cfg.num_workers, sampler)
    val_loader = create_loader(val_ds, cfg.batch_size, False, cfg.num_workers)

    model = build_task3_model(num_classes=len(label_map), pretrained=True).to(device)
    model, history = fit_model(model, train_loader, val_loader, train_df, device, cfg)
    torch.save(model.state_dict(), task_dir / "task3_best.pt")
    plot_history(history, "Task 3", task_dir / "task3_history.png")
    save_json(history, task_dir / "task3_history.json")

    crop_candidates = [1, 3]
    val_df_res, val_out_by_crop = run_task3_crop_sweep(
        model, val_df, device, clip_len, sample_rate, img_size, crop_candidates
    )
    test_df_res, test_out_by_crop = run_task3_crop_sweep(
        model, test_df, device, clip_len, sample_rate, img_size, crop_candidates
    )
    save_dataframe(val_df_res, task_dir / "task3_val_crop_compare.csv")
    save_dataframe(test_df_res, task_dir / "task3_test_crop_compare.csv")

    best_row = val_df_res.sort_values(["macro_f1", "accuracy"], ascending=False).iloc[0]
    best_crops = int(best_row["num_temporal_crops"])
    best_out = test_out_by_crop[best_crops]

    report_df = classification_report_dataframe(best_out["y_true"], best_out["y_pred"], idx_to_class)
    pred_df = save_prediction_table(test_df, best_out["y_true"], best_out["y_pred"], idx_to_class)
    save_dataframe(report_df, task_dir / "task3_classification_report.csv")
    save_dataframe(pred_df, task_dir / "task3_test_predictions.csv")
    plot_confusion_matrix(
        best_out["y_true"],
        best_out["y_pred"],
        idx_to_class,
        f"Task 3 Confusion Matrix ({best_crops} crop)",
        task_dir / "task3_confusion.png",
    )

    summary = {
        "task": "Task3",
        "model": f"Task3_3DCNN_{best_crops}crop",
        "accuracy": float(best_out["accuracy"]),
        "macro_f1": float(best_out["macro_f1"]),
        "avg_sec_per_video": float(best_out["avg_sec_per_video"]),
        "best_num_temporal_crops_by_val": best_crops,
    }
    save_json(summary, task_dir / "task3_summary.json")
    print(summary)
    return summary


def run_task4(
    split_root: Path,
    output_root: Path,
    device: torch.device,
    img_size: int = 112,
    seq_len: int = 12,
    base_stride: int = 2,
    encoder_tune_mode: str = "partial",
) -> Dict[str, object]:
    train_df, val_df, test_df, label_map = load_splits(split_root)
    idx_to_class = {v: k for k, v in label_map.items()}
    suffix = f"task4_{encoder_tune_mode}"
    task_dir = ensure_dir(output_root / suffix)

    cfg = TrainConfig(
        epochs=25,
        batch_size=8,
        num_workers=2,
        lr=5e-4,
        weight_decay=1e-4,
        label_smoothing=0.05,
        early_stop_patience=6,
        use_weighted_sampler=True,
        use_amp=True,
    )
    head_lr = 5e-4
    encoder_lr = 1e-4 if encoder_tune_mode != "full" else 5e-5
    selection_seq_candidates = [1, 3]
    selection_num_repeats = 3

    train_ds = SequenceDatasetLSTM(train_df, seq_len=seq_len, img_size=img_size, training=True, base_stride=base_stride)
    val_ds = SequenceDatasetLSTM(val_df, seq_len=seq_len, img_size=img_size, training=False, base_stride=base_stride)
    sampler = build_weighted_sampler(train_df) if cfg.use_weighted_sampler else None
    train_loader = create_loader(train_ds, cfg.batch_size, True, cfg.num_workers, sampler)
    val_loader = create_loader(val_ds, cfg.batch_size, False, cfg.num_workers)

    model = ResNetBiLSTMAttention(
        num_classes=len(label_map),
        hidden_size=128,
        num_layers=1,
        dropout=0.4,
        bidirectional=True,
        encoder_tune_mode=encoder_tune_mode,
    ).to(device)
    model, history, best_num_sequences = fit_task4_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_df=train_df,
        val_df=val_df,
        cfg=cfg,
        device=device,
        seq_len=seq_len,
        img_size=img_size,
        base_stride=base_stride,
        selection_seq_candidates=selection_seq_candidates,
        selection_num_repeats=selection_num_repeats,
        head_lr=head_lr,
        encoder_lr=encoder_lr,
    )
    torch.save(model.state_dict(), task_dir / f"{suffix}_best.pt")
    plot_history(history, f"Task 4 ({encoder_tune_mode})", task_dir / f"{suffix}_history.png")
    save_json(history, task_dir / f"{suffix}_history.json")

    val_df_res, val_out_by_seq = run_task4_sequence_sweep(
        model,
        val_df,
        device,
        seq_len,
        img_size,
        base_stride,
        selection_seq_candidates,
        num_repeats=selection_num_repeats,
    )
    test_df_res, test_out_by_seq = run_task4_sequence_sweep(
        model,
        test_df,
        device,
        seq_len,
        img_size,
        base_stride,
        selection_seq_candidates,
        num_repeats=selection_num_repeats,
    )
    save_dataframe(val_df_res, task_dir / f"{suffix}_val_sequence_compare.csv")
    save_dataframe(test_df_res, task_dir / f"{suffix}_test_sequence_compare.csv")

    best_out = test_out_by_seq[int(best_num_sequences)]
    report_df = classification_report_dataframe(best_out["y_true"], best_out["y_pred"], idx_to_class)
    pred_df = save_prediction_table(test_df, best_out["y_true"], best_out["y_pred"], idx_to_class)
    save_dataframe(report_df, task_dir / f"{suffix}_classification_report.csv")
    save_dataframe(pred_df, task_dir / f"{suffix}_test_predictions.csv")
    plot_confusion_matrix(
        best_out["y_true"],
        best_out["y_pred"],
        idx_to_class,
        f"Task 4 Confusion Matrix ({encoder_tune_mode}, {best_num_sequences} sequences, {selection_num_repeats} repeats)",
        task_dir / f"{suffix}_confusion.png",
    )

    summary = {
        "task": "Task4",
        "model": f"Task4_ResNet18_BiLSTM_Attn_{encoder_tune_mode}_seq{seq_len}_{best_num_sequences}seq_{selection_num_repeats}rep",
        "accuracy": float(best_out["accuracy"]),
        "macro_f1": float(best_out["macro_f1"]),
        "avg_sec_per_video": float(best_out["avg_sec_per_video"]),
        "best_num_sequences_by_val": int(best_num_sequences),
        "selection_num_repeats": int(selection_num_repeats),
        "encoder_tune_mode": encoder_tune_mode,
        "seq_len": seq_len,
        "base_stride": base_stride,
    }
    save_json(summary, task_dir / f"{suffix}_summary.json")
    print(summary)
    return summary


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------

def run_all(args: argparse.Namespace) -> None:
    device = resolve_device()
    print("Using device:", device)

    data_root = Path(args.data_root).resolve()
    split_root = ensure_dir(Path(args.split_root).resolve())
    output_root = ensure_dir(Path(args.output_root).resolve())

    print_runtime_paths(data_root, split_root, output_root)

    if args.make_splits or not (split_root / "train.csv").exists():
        make_splits(data_root, split_root)

    results: List[Dict[str, object]] = []
    results.append(run_task2(split_root, output_root, device, img_size=args.img_size))
    results.append(
        run_task3(
            split_root,
            output_root,
            device,
            img_size=args.img_size,
            clip_len=args.clip_len,
            sample_rate=args.sample_rate,
        )
    )
    results.append(
        run_task4(
            split_root,
            output_root,
            device,
            img_size=args.img_size,
            seq_len=args.seq_len,
            base_stride=args.base_stride,
            encoder_tune_mode=args.encoder_tune_mode,
        )
    )

    results_df = pd.DataFrame(results)
    save_dataframe(results_df, output_root / "all_results_summary.csv")
    plot_results_summary(results_df, output_root / "all_results_summary.png")
    print(results_df)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UCF11 HW3 training pipeline for Colab + GitHub.")
    parser.add_argument("--task", type=str, default="all", choices=["make_splits", "task2", "task3", "task4", "all"])
    parser.add_argument("--data_root", type=str, default=str(DATA_ROOT))
    parser.add_argument("--split_root", type=str, default=str(SPLIT_ROOT))
    parser.add_argument("--output_root", type=str, default=str(OUTPUT_ROOT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img_size", type=int, default=112)
    parser.add_argument("--clip_len", type=int, default=16)
    parser.add_argument("--sample_rate", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--base_stride", type=int, default=2)
    parser.add_argument("--encoder_tune_mode", type=str, default="partial", choices=["freeze", "partial", "full"])
    parser.add_argument("--make_splits", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = resolve_device()
    print("Using device:", device)

    data_root = Path(args.data_root).resolve()
    split_root = ensure_dir(Path(args.split_root).resolve())
    output_root = ensure_dir(Path(args.output_root).resolve())

    print_runtime_paths(data_root, split_root, output_root)

    if args.task == "make_splits":
        make_splits(data_root, split_root)
        return

    if args.make_splits or not (split_root / "train.csv").exists():
        make_splits(data_root, split_root)

    if args.task == "task2":
        run_task2(split_root, output_root, device, img_size=args.img_size)
    elif args.task == "task3":
        run_task3(
            split_root,
            output_root,
            device,
            img_size=args.img_size,
            clip_len=args.clip_len,
            sample_rate=args.sample_rate,
        )
    elif args.task == "task4":
        run_task4(
            split_root,
            output_root,
            device,
            img_size=args.img_size,
            seq_len=args.seq_len,
            base_stride=args.base_stride,
            encoder_tune_mode=args.encoder_tune_mode,
        )
    else:
        run_all(args)


if __name__ == "__main__":
    main()
