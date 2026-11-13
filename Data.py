from pathlib import Path
from pathlib import PurePosixPath
from statistics import pstdev

import numpy as np
import pandas as pd


DATASET_HANDLE = "ashery/chexpert"
TSNE_SAMPLE_SIZE = None
TSNE_RANDOM_STATE = 42
FINDING_LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Lesion",
    "Lung Opacity",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


def load_labeled_images(dataset_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    metadata_frames = []
    finding_labels = set()

    for csv_path in sorted(dataset_dir.glob("*.csv")):
        columns = pd.read_csv(csv_path, nrows=0).columns
        labels_in_file = [label for label in FINDING_LABELS if label in columns]
        path_column = next(
            (column for column in columns if column.lower() in {"path", "image_path"}),
            None,
        )
        if not labels_in_file or path_column is None:
            continue

        frame = pd.read_csv(csv_path, usecols=[path_column, *labels_in_file])
        frame = frame.rename(columns={path_column: "Path"})
        metadata_frames.append(frame)
        finding_labels.update(labels_in_file)

    if not metadata_frames:
        raise FileNotFoundError(
            f"No labeled image CSV files with a Path column were found under {dataset_dir}"
        )

    images = pd.concat(metadata_frames, ignore_index=True)
    images["Path"] = images["Path"].astype("string").str.replace("\\", "/", regex=False)
    images = images.dropna(subset=["Path"])

    labels = [label for label in FINDING_LABELS if label in finding_labels]
    for label in labels:
        images[label] = pd.to_numeric(images[label], errors="coerce")
    duplicate_paths = images[images["Path"].duplicated(keep=False)]
    for path, rows in duplicate_paths.groupby("Path", sort=False):
        conflicts = [
            label for label in labels if rows[label].dropna().nunique() > 1
        ]
        if conflicts:
            raise ValueError(
                f"Conflicting labels for image {path}: {', '.join(conflicts)}"
            )
    images = images.groupby("Path", as_index=False, sort=False)[labels].first()
    return images, labels


def build_report(images: pd.DataFrame, labels: list[str], dataset_dir: Path) -> str:
    total_images = len(images)
    status_columns = {
        "negative": "Negative (0)",
        "positive": "Positive (1)",
        "uncertain": "Uncertain (-1)",
        "missing": "Missing",
    }
    counts: dict[str, dict[str, int]] = {}
    percentages: dict[str, dict[str, float]] = {}
    for label in labels:
        values = images[label]
        counts[label] = {
            "negative": int((values == 0).sum()),
            "positive": int((values == 1).sum()),
            "uncertain": int((values == -1).sum()),
            "missing": int(values.isna().sum()),
        }
        percentages[label] = {
            status: (count / total_images * 100 if total_images else 0)
            for status, count in counts[label].items()
        }

    imbalance_scores = {}
    for label in labels:
        status_percentages = [
            percentages[label][status] for status in status_columns
        ]
        mean_percentage = sum(status_percentages) / len(status_percentages)
        imbalance_scores[label] = (
            pstdev(status_percentages) / mean_percentage
            if mean_percentage
            else 0
        )
    ranked_imbalance = sorted(
        labels,
        key=lambda label: imbalance_scores[label],
        reverse=True,
    )

    lines = [
        "# CheXpert Dataset Analysis Report",
        "",
        f"- Dataset: `{DATASET_HANDLE}`",
        f"- Dataset directory: `{dataset_dir}`",
        f"- Number of labeled classes: **{len(labels)}**",
        f"- Total unique images: **{total_images:,}**",
        "",
        "## Number of Images per Class",
        "",
        "| Class | "
        + " | ".join(
            f"{english} count (% of total)"
            for english in status_columns.values()
        )
        + " |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in labels:
        status_cells = [
            f"{counts[label][status]:,} ({percentages[label][status]:.2f}%)"
            for status in status_columns
        ]
        lines.append(f"| {label} | " + " | ".join(status_cells) + " |")

    lines.extend(
        [
            "",
            "## Class Imbalance",
            "",
            "Ranked by the coefficient of variation across the four label-status percentages; "
            "a higher score indicates a more imbalanced distribution.",
            "",
            "| Rank | Finding | "
            + " | ".join(
                f"{english} %"
                for english in status_columns.values()
            )
            + " | Imbalance score |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, label in enumerate(ranked_imbalance, start=1):
        status_cells = [
            f"{percentages[label][status]:.2f}%"
            for status in status_columns
        ]
        lines.append(
            f"| {rank} | {label} | "
            + " | ".join(status_cells)
            + f" | {imbalance_scores[label]:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Clinical Findings Notes",
            "",
            "- Labels are extracted from radiology reports and are not independently verified patient diagnoses.",
            "- An image may have multiple findings, so positive image sets can overlap.",
            "- `1` means positive, `0` means negative, and `-1` means uncertain; missing labels are reported separately.",
            "- This report describes dataset labels only; it does not establish clinical prevalence, causation, or diagnostic performance.",
        ]
    )
    lines.append("")
    return "\n".join(lines)


def resolve_image_path(image_path: str, dataset_dir: Path) -> Path:
    parts = PurePosixPath(image_path.replace("\\", "/")).parts
    partition_index = next(
        (
            index
            for index, part in enumerate(parts)
            if part.lower() in {"train", "valid"}
        ),
        None,
    )
    if partition_index is None:
        raise ValueError(
            f"Image path does not contain a train/valid directory: {image_path}"
        )
    return dataset_dir.joinpath(*parts[partition_index:])


def create_tsne_visualization(
    images: pd.DataFrame,
    labels: list[str],
    dataset_dir: Path,
    output_path: Path,
    sample_size: int | None = TSNE_SAMPLE_SIZE,
) -> int:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from PIL import Image
        from sklearn.decomposition import IncrementalPCA
        from sklearn.manifold import TSNE
    except ImportError as error:
        raise RuntimeError(
            "t-SNE visualization requires matplotlib, Pillow, scikit-learn, "
            "and a compatible SciPy installation."
        ) from error

    sampled_images = (
        images
        if sample_size is None or sample_size >= len(images)
        else images.sample(n=sample_size, random_state=TSNE_RANDOM_STATE)
    )
    feature_vectors = np.empty((len(sampled_images), 32 * 32), dtype=np.uint8)
    class_values = np.empty((len(sampled_images), len(labels)), dtype=np.float32)
    path_index = sampled_images.columns.get_loc("Path")
    label_indices = [sampled_images.columns.get_loc(label) for label in labels]
    for index, row in enumerate(sampled_images.itertuples(index=False, name=None)):
        image_path = resolve_image_path(str(row[path_index]), dataset_dir)
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Dataset is incomplete: sampled image {image_path} does not exist."
            )
        with Image.open(image_path) as image:
            grayscale = image.convert("L").resize((32, 32), Image.Resampling.BILINEAR)
            feature_vectors[index] = np.asarray(grayscale, dtype=np.uint8).reshape(-1)
        class_values[index] = [row[label_index] for label_index in label_indices]

    features = feature_vectors.astype(np.float32)
    image_count = len(features)
    feature_count = features.shape[1]
    del feature_vectors
    features /= 255.0
    pca_components = min(50, features.shape[0] - 1, features.shape[1])
    reduced_features = IncrementalPCA(
        n_components=pca_components,
        batch_size=max(pca_components, min(4096, features.shape[0])),
    ).fit_transform(features)
    del features
    perplexity = min(30.0, (image_count - 1) / 3)
    coordinates = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        n_jobs=-1,
        random_state=TSNE_RANDOM_STATE,
    ).fit_transform(reduced_features)

    statuses = [
        ("missing", None, "Missing", "#B8B8B8", 2, 0.16),
        ("negative", 0, "Negative (0)", "#0072B2", 3, 0.75),
        ("uncertain", -1, "Uncertain (-1)", "#E69F00", 4, 0.9),
        ("positive", 1, "Positive (1)", "#D55E00", 5, 1.0),
    ]
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = plt.subplots(2, 7, figsize=(25, 9), sharex=True, sharey=True)
    for label_index, (axis, label) in enumerate(zip(axes.flat, labels)):
        values = class_values[:, label_index]
        for status, value, _, color, marker_size, alpha in statuses:
            mask = np.isnan(values) if value is None else values == value
            if np.any(mask):
                axis.scatter(
                    coordinates[mask, 0],
                    coordinates[mask, 1],
                    s=marker_size,
                    alpha=alpha,
                    color=color,
                    linewidths=0,
                    rasterized=True,
                )
        axis.set_title(f"{label}\nFinding label", fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xlabel("t-SNE 1")
        axis.set_ylabel("t-SNE 2")

    legend_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=7,
            markerfacecolor=color,
            markeredgewidth=0,
            label=legend_label,
        )
        for _, _, legend_label, color, _, _ in statuses
    ]
    figure.suptitle(
        "Chest X-ray Image t-SNE",
        fontsize=16,
    )
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.065),
        frameon=False,
    )
    figure.text(
        0.5,
        0.025,
        f"Grayscale images are resized to 32×32 ({feature_count:,} pixel features); "
        f"each panel is colored by its finding label; {image_count:,} images used.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.12, 1, 0.94))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return image_count


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    dataset_output_dir = project_dir / "dataset"
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import kagglehub
    except ImportError as error:
        raise RuntimeError(
            "KaggleHub is required. Install it with `python -m pip install kagglehub`."
        ) from error

    try:
        dataset_dir = Path(
            kagglehub.dataset_download(
                DATASET_HANDLE,
                output_dir=str(dataset_output_dir),
            )
        )
    except FileExistsError:
        dataset_dir = Path(
            kagglehub.dataset_download(
                DATASET_HANDLE,
                output_dir=str(dataset_output_dir),
                force_download=True,
            )
        )
    images, labels = load_labeled_images(dataset_dir)
    tsne_path = project_dir / "chexpert_tsne.png"
    tsne_image_count = create_tsne_visualization(
        images,
        labels,
        dataset_dir,
        tsne_path,
    )
    report_path = project_dir / "chexpert_analysis.md"
    report = build_report(images, labels, dataset_dir)
    report += (
        "\n## t-SNE Visualization\n\n"
        f"- Figure: [{tsne_path.name}](./{tsne_path.name})\n"
        f"- Images used: **{tsne_image_count:,}**\n"
        "- Method: Uses all labeled images; grayscale images are resized to 32×32 "
        "(1,024 pixel features per image), then reduced with incremental PCA and t-SNE; "
        "each panel shows one finding's label status.\n"
    )
    report_path.write_text(report, encoding="utf-8")
    print(f"Analyzed {len(images):,} unique images across {len(labels)} classes.")
    print(f"Report written to: {report_path}")
    print(f"t-SNE visualization written to: {tsne_path}")


if __name__ == "__main__":
    main()