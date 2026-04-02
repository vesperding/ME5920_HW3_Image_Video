
from pathlib import Path
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path("/content/ME5920_HW3_Image_Video")
OUTPUT_ROOT = REPO_ROOT / "outputs"
SPLIT_ROOT = REPO_ROOT / "splits"
FIG_ROOT = OUTPUT_ROOT / "figures"
FIG_ROOT.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.dpi"] = 140


def save_fig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_tradeoff(val_csv, test_csv, x_col, title_prefix, save_path):
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(val_df[x_col], val_df["accuracy"], marker="o", label="val_acc")
    axes[0].plot(test_df[x_col], test_df["accuracy"], marker="o", label="test_acc")
    axes[0].set_title(f"{title_prefix} Accuracy")
    axes[0].set_xlabel(x_col)
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(val_df[x_col], val_df["macro_f1"], marker="o", label="val_macro_f1")
    axes[1].plot(test_df[x_col], test_df["macro_f1"], marker="o", label="test_macro_f1")
    axes[1].set_title(f"{title_prefix} Macro-F1")
    axes[1].set_xlabel(x_col)
    axes[1].set_ylabel("Macro-F1")
    axes[1].legend()

    if "avg_sec_per_video" in test_df.columns:
        axes[2].plot(test_df[x_col], test_df["avg_sec_per_video"], marker="o", label="test_time")
        axes[2].set_title(f"{title_prefix} Inference Time")
        axes[2].set_xlabel(x_col)
        axes[2].set_ylabel("Avg sec / video")
        axes[2].legend()
    else:
        axes[2].axis("off")

    save_fig(fig, save_path)


def resolve_f1_column(df):
    for cand in ["f1", "f1-score", "macro_f1"]:
        if cand in df.columns:
            return cand
    raise KeyError(f"Cannot find F1 column. Available columns: {list(df.columns)}")


def plot_per_class_f1(report_csv, title, save_path):
    df = pd.read_csv(report_csv)
    f1_col = resolve_f1_column(df)

    if "class_name" not in df.columns:
        raise KeyError(f"'class_name' not found in {report_csv}. Available columns: {list(df.columns)}")

    df = df[~df["class_name"].astype(str).isin(["accuracy", "macro avg", "weighted avg"])]
    df[f1_col] = pd.to_numeric(df[f1_col], errors="coerce")
    df = df.dropna(subset=[f1_col]).sort_values(f1_col, ascending=False)

    fig = plt.figure(figsize=(12, 5))
    plt.bar(df["class_name"], df[f1_col])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("F1-score")
    plt.title(title)

    save_fig(fig, save_path)


def sample_four_frames(video_path, n_frames=4, img_size=224):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        return [np.zeros((img_size, img_size, 3), dtype=np.uint8) for _ in range(n_frames)]

    idxs = np.linspace(0, max(total - 1, 0), n_frames).astype(int).tolist()
    frames = []

    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            frame = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (img_size, img_size))
        frames.append(frame)

    cap.release()
    return frames


def resolve_prediction_columns(df):
    true_name_col = None
    for c in ["true_name", "class_name", "true_label"]:
        if c in df.columns:
            true_name_col = c
            break

    pred_name_col = None
    for c in ["pred_name", "pred_label"]:
        if c in df.columns:
            pred_name_col = c
            break

    video_col = None
    for c in ["video_path", "path"]:
        if c in df.columns:
            video_col = c
            break

    return true_name_col, pred_name_col, video_col


def plot_prediction_examples(pred_csv, title, save_path, n_correct=2, n_wrong=4, img_size=160, seed=42):
    random.seed(seed)
    df = pd.read_csv(pred_csv)

    true_name_col, pred_name_col, video_col = resolve_prediction_columns(df)

    assert true_name_col is not None, f"Cannot find GT column in {pred_csv}. Columns: {list(df.columns)}"
    assert pred_name_col is not None, f"Cannot find prediction column in {pred_csv}. Columns: {list(df.columns)}"
    assert video_col is not None, f"Cannot find video path column in {pred_csv}. Columns: {list(df.columns)}"

    if "correct" not in df.columns:
        df["correct"] = df[true_name_col].astype(str) == df[pred_name_col].astype(str)

    correct_df = df[df["correct"] == True].copy()
    wrong_df = df[df["correct"] == False].copy()

    selected_parts = []
    if len(wrong_df) > 0:
        selected_parts.append(wrong_df.sample(min(n_wrong, len(wrong_df)), random_state=seed))
    if len(correct_df) > 0:
        selected_parts.append(correct_df.sample(min(n_correct, len(correct_df)), random_state=seed))

    if len(selected_parts) == 0:
        print(f"Skip prediction examples for {pred_csv}: no usable rows found.")
        return

    selected_df = pd.concat(selected_parts, axis=0).reset_index(drop=True)

    n_rows = len(selected_df)
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.8 * n_rows))

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, row in selected_df.iterrows():
        video_path = row[video_col]
        gt = row[true_name_col]
        pred = row[pred_name_col]
        ok = bool(row["correct"])

        frames = sample_four_frames(video_path, n_frames=4, img_size=img_size)

        for j in range(n_cols):
            ax = axes[i, j]
            ax.imshow(frames[j])
            ax.axis("off")
            if j == 0:
                ax.set_title(f"{'Correct' if ok else 'Wrong'}\nGT: {gt}\nPred: {pred}", fontsize=10)

    fig.suptitle(title, fontsize=14)
    save_fig(fig, save_path)


def plot_overall_summary(summary_csv, save_path):
    df = pd.read_csv(summary_csv)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].bar(df["task"], df["accuracy"])
    axes[0].set_title("Overall Accuracy")
    axes[0].set_ylabel("Accuracy")

    axes[1].bar(df["task"], df["macro_f1"])
    axes[1].set_title("Overall Macro-F1")
    axes[1].set_ylabel("Macro-F1")

    if "avg_sec_per_video" in df.columns:
        axes[2].bar(df["task"], df["avg_sec_per_video"])
        axes[2].set_title("Overall Inference Time")
        axes[2].set_ylabel("Avg sec / video")
    else:
        axes[2].axis("off")

    save_fig(fig, save_path)


def plot_dataset_split_distribution(split_root, save_path_png, save_path_csv):
    train_df = pd.read_csv(split_root / "train.csv")
    val_df = pd.read_csv(split_root / "val.csv")
    test_df = pd.read_csv(split_root / "test.csv")

    train_count = train_df.groupby("class_name").size().rename("train")
    val_count = val_df.groupby("class_name").size().rename("val")
    test_count = test_df.groupby("class_name").size().rename("test")

    split_count_df = pd.concat([train_count, val_count, test_count], axis=1).fillna(0).astype(int)
    split_count_df = split_count_df.reset_index()

    fig = plt.figure(figsize=(14, 6))
    x = range(len(split_count_df))

    plt.bar([i - 0.25 for i in x], split_count_df["train"], width=0.25, label="train")
    plt.bar(x, split_count_df["val"], width=0.25, label="val")
    plt.bar([i + 0.25 for i in x], split_count_df["test"], width=0.25, label="test")

    plt.xticks(list(x), split_count_df["class_name"], rotation=35, ha="right")
    plt.ylabel("Number of videos")
    plt.title("Dataset Split Distribution by Class")
    plt.legend()

    save_fig(fig, save_path_png)
    split_count_df.to_csv(save_path_csv, index=False)
    print(f"Saved: {save_path_csv}")


def main():
    plot_dataset_split_distribution(
        SPLIT_ROOT,
        FIG_ROOT / "dataset_split_distribution.png",
        FIG_ROOT / "dataset_split_distribution.csv",
    )

    plot_tradeoff(
        OUTPUT_ROOT / "task2" / "task2_val_frame_compare.csv",
        OUTPUT_ROOT / "task2" / "task2_test_frame_compare.csv",
        x_col="n_frames",
        title_prefix="Task2: Random Frames",
        save_path=FIG_ROOT / "task2_tradeoff.png",
    )
    plot_per_class_f1(
        OUTPUT_ROOT / "task2" / "task2_classification_report.csv",
        title="Task2 Per-Class F1",
        save_path=FIG_ROOT / "task2_per_class_f1.png",
    )
    plot_prediction_examples(
        OUTPUT_ROOT / "task2" / "task2_test_predictions.csv",
        title="Task2 Prediction Examples",
        save_path=FIG_ROOT / "task2_prediction_examples.png",
        n_correct=2,
        n_wrong=4,
    )

    plot_tradeoff(
        OUTPUT_ROOT / "task3" / "task3_val_crop_compare.csv",
        OUTPUT_ROOT / "task3" / "task3_test_crop_compare.csv",
        x_col="num_temporal_crops",
        title_prefix="Task3: Temporal Crops",
        save_path=FIG_ROOT / "task3_tradeoff.png",
    )
    plot_per_class_f1(
        OUTPUT_ROOT / "task3" / "task3_classification_report.csv",
        title="Task3 Per-Class F1",
        save_path=FIG_ROOT / "task3_per_class_f1.png",
    )
    plot_prediction_examples(
        OUTPUT_ROOT / "task3" / "task3_test_predictions.csv",
        title="Task3 Prediction Examples",
        save_path=FIG_ROOT / "task3_prediction_examples.png",
        n_correct=2,
        n_wrong=4,
    )

    plot_tradeoff(
        OUTPUT_ROOT / "task4_partial" / "task4_partial_val_sequence_compare.csv",
        OUTPUT_ROOT / "task4_partial" / "task4_partial_test_sequence_compare.csv",
        x_col="num_sequences",
        title_prefix="Task4: Sequence Count",
        save_path=FIG_ROOT / "task4_tradeoff.png",
    )
    plot_per_class_f1(
        OUTPUT_ROOT / "task4_partial" / "task4_partial_classification_report.csv",
        title="Task4 Per-Class F1",
        save_path=FIG_ROOT / "task4_per_class_f1.png",
    )
    plot_prediction_examples(
        OUTPUT_ROOT / "task4_partial" / "task4_partial_test_predictions.csv",
        title="Task4 Prediction Examples",
        save_path=FIG_ROOT / "task4_prediction_examples.png",
        n_correct=2,
        n_wrong=4,
    )

    plot_overall_summary(
        OUTPUT_ROOT / "all_results_summary.csv",
        save_path=FIG_ROOT / "overall_summary_barplots.png",
    )

    print("\nGenerated figure files:")
    for p in sorted(FIG_ROOT.glob("*")):
        print(p)


if __name__ == "__main__":
    main()
