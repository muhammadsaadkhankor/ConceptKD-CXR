"""Full metadata analysis for five CheXpert findings; no image pixels are loaded."""

from __future__ import annotations

from itertools import combinations
from os.path import isfile
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Change only these paths if this project is moved. None means the full CSV.
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"
TRAIN_CSV_PATH = DATASET_DIR / "train.csv"
IMAGE_DIR = DATASET_DIR / "train"
OUTPUT_DIR = PROJECT_DIR / "chexpert_full_analysis"
SAMPLE_SIZE: int | None = None

DISEASES = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
LABEL_COLUMNS = ["Positive (1)", "Negative (0)", "Uncertain (-1)", "Missing (NaN)"]
LABEL_COLORS = ["#2a9d8f", "#457b9d", "#f4a261", "#bdbdbd"]


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without an optional dependency."""
    if frame.empty:
        return "_No rows._"
    columns = [str(column) for column in frame.columns]
    rows = frame.fillna("").astype(str).values.tolist()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend("| " + " | ".join(value.replace("|", "\\|") for value in row) + " |" for row in rows)
    return "\n".join(lines)


def write_workbook(sections: dict[str, list[tuple[str, pd.DataFrame]]]) -> Path:
    """Store all tables in one Excel workbook, with related tables sharing sheets."""
    workbook = OUTPUT_DIR / "chexpert_analysis_tables.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        for sheet_name, blocks in sections.items():
            row = 0
            for title, table in blocks:
                pd.DataFrame([[title]]).to_excel(writer, sheet_name=sheet_name, startrow=row, index=False, header=False)
                table.to_excel(writer, sheet_name=sheet_name, startrow=row + 1, index=False)
                row += len(table) + 4
            sheet = writer.sheets[sheet_name]
            sheet.freeze_panes = "A2"
            for cells in sheet.columns:
                width = max((len(str(cell.value or "")) for cell in cells), default=10) + 2
                sheet.column_dimensions[cells[0].column_letter].width = min(max(width, 12), 58)
    return workbook


def patient_id(path: object) -> str | float:
    match = re.search(r"(?:^|[\\/])(patient[^\\/]+)(?:[\\/]|$)", str(path), flags=re.I)
    return match.group(1) if match else np.nan


def image_file_check(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check paths only when the local training-image directory exists."""
    columns = ["Check", "Status", "Value", "Details"]
    if "Path" not in data:
        return pd.DataFrame([["Image-file existence", "Not completed", np.nan, "CSV has no Path column."]], columns=columns), pd.DataFrame(columns=["Path", "Reason"])
    if not IMAGE_DIR.is_dir():
        detail = f"Expected image directory is unavailable: {IMAGE_DIR}"
        return pd.DataFrame([["Image-file existence", "Not completed", np.nan, detail]], columns=columns), pd.DataFrame(columns=["Path", "Reason"])

    raw_paths = data["Path"].astype("string")
    normalized = raw_paths.str.replace("\\", "/", regex=False)
    relative_paths = normalized.str.extract(r"(?:^|/)train/(.*)$", expand=False)
    valid_path = relative_paths.notna() & relative_paths.ne("")
    exists = pd.Series(False, index=data.index)
    # This checks filenames only; it does not open or decode image contents.
    exists.loc[valid_path] = [isfile(IMAGE_DIR / relative) for relative in relative_paths.loc[valid_path]]
    missing_mask = ~exists
    missing = pd.DataFrame(
        {
            "Path": data.loc[missing_mask, "Path"].to_numpy(),
            "Reason": np.where(valid_path.loc[missing_mask], "File not found", "Blank or unrecognised CSV path"),
        }
    )
    summary = pd.DataFrame(
        [
            ["Image-file existence", "Completed", len(data), "Image paths checked; image pixels were not loaded."],
            ["Missing image files", "Completed", len(missing), "Reported separately from missing disease labels."],
        ],
        columns=columns,
    )
    return summary, missing


def concept_mapping() -> pd.DataFrame:
    """Educational imaging concepts, deliberately separate from CheXpert labels."""
    concepts = [
        ("Atelectasis", "Volume loss with linear or plate-like opacity", "Underexpanded lung may appear linear, wedge-shaped, or plate-like with reduced regional volume."),
        ("Atelectasis", "Fissural or mediastinal displacement", "Volume loss can displace fissures, the hilum, mediastinum, or hemidiaphragm toward the affected region."),
        ("Cardiomegaly", "Enlarged cardiac silhouette", "A widened heart silhouette can be visible on frontal radiography; projection and patient position affect apparent size."),
        ("Cardiomegaly", "Cardiothoracic ratio", "On a properly positioned upright PA image, a ratio above about 0.5 is a conventional screening sign."),
        ("Consolidation", "Air-space opacity", "Alveolar filling can produce a confluent, lobar, or segmental pulmonary opacity."),
        ("Consolidation", "Air bronchograms", "Air-filled bronchi outlined by surrounding air-space opacity support an air-space process."),
        ("Edema", "Bilateral perihilar or diffuse opacity", "Hydrostatic pulmonary edema can produce bilateral central or diffuse hazy air-space opacities."),
        ("Edema", "Interstitial edema signs", "Septal thickening, including Kerley B lines, and vascular redistribution are established radiographic findings."),
        ("Pleural Effusion", "Costophrenic-angle blunting and meniscus", "Pleural fluid can blunt a costophrenic angle and form a concave upper fluid margin."),
        ("Pleural Effusion", "Dependent layering", "Free pleural fluid can layer dependently; a decubitus image may demonstrate mobility."),
    ]
    return pd.DataFrame(concepts, columns=["Disease label", "Radiological concept", "Brief explanation"]).assign(
        **{
            "Relationship to dataset label": "Imaging concept only; never transforms or replaces the CSV label.",
            "Literature verification status": "Verify before research publication or clinical interpretation.",
        }
    )


def save_figures(label_summary: pd.DataFrame, matrix: pd.DataFrame, view_counts: pd.DataFrame, view_labels: pd.DataFrame) -> None:
    """Create the requested distribution, co-occurrence, and view figures."""
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(label_summary), dtype=int)
    for column, color in zip(LABEL_COLUMNS, LABEL_COLORS):
        values = label_summary[column].to_numpy()
        ax.bar(label_summary["Disease"], values, bottom=bottom, label=column, color=color)
        bottom += values
    ax.set(title="Label distribution by disease", ylabel="Image records")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="CSV label state", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "label_distribution_stacked_bar.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", square=True, cbar_kws={"label": "Image records"}, ax=ax)
    ax.set(title="Positive-label disease co-occurrence", xlabel="", ylabel="")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "disease_cooccurrence_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    if not view_counts.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=view_counts, x="View", y="Image records", hue="View", legend=False, palette="viridis", ax=ax)
        ax.set(title="Frontal and lateral image records")
        ax.bar_label(ax.containers[0])
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "frontal_lateral_counts.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    if not view_labels.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=view_labels, x="Disease", y="Positive percentage", hue="View", ax=ax)
        ax.set(title="Positive-label distribution by view", ylabel="Positive labels (% of images in view)")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "frontal_lateral_positive_label_comparison.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def write_report(
    dataset_stats: pd.DataFrame,
    label_summary: pd.DataFrame,
    positive_counts: pd.DataFrame,
    imbalance: pd.DataFrame,
    view_counts: pd.DataFrame,
    image_summary: pd.DataFrame,
    status: pd.DataFrame,
    at_least_two: int,
    top_pair: pd.Series | None,
) -> None:
    pair_text = "No disease-pair rows were available."
    if top_pair is not None:
        pair_text = f"Largest positive pair: **{top_pair['Disease 1']} + {top_pair['Disease 2']}** ({top_pair['Co-occurrence count (both positive)']} records)."
    report = [
        "# CheXpert five-disease analysis",
        "",
        "## Scope",
        "",
        f"Analyzed **{dataset_stats.loc[0, 'Value']}** training records. `SAMPLE_SIZE = {SAMPLE_SIZE!r}`; `None` denotes the full training CSV.",
        "No image pixels were loaded and source dataset files were not modified. Only label value `1` is positive; `0`, `-1`, and missing values remain distinct.",
        "All tables are consolidated in `chexpert_analysis_tables.xlsx`.",
        "",
        "## Dataset statistics",
        "",
        markdown_table(dataset_stats),
        "",
        "## Label distribution",
        "",
        markdown_table(label_summary),
        "",
        "## Co-occurrence",
        "",
        markdown_table(positive_counts),
        "",
        f"Images with at least two positive labels: **{at_least_two}**. {pair_text}",
        "",
        "## Class imbalance (known binary labels only)",
        "",
        markdown_table(imbalance),
        "",
        "## Frontal/lateral views",
        "",
        markdown_table(view_counts),
        "",
        "## Image-file check",
        "",
        markdown_table(image_summary),
        "",
        "## Workflow status",
        "",
        markdown_table(status),
        "",
        "Radiological concepts in the workbook are educational descriptors, separate from CheXpert labels, and marked for literature verification.",
    ]
    (OUTPUT_DIR / "summary_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not TRAIN_CSV_PATH.is_file():
        raise FileNotFoundError(f"Training CSV not found: {TRAIN_CSV_PATH}")

    read_options = {} if SAMPLE_SIZE is None else {"nrows": SAMPLE_SIZE}
    data = pd.read_csv(TRAIN_CSV_PATH, **read_options)
    warnings: list[str] = []

    labels = pd.DataFrame(index=data.index)
    for disease in DISEASES:
        if disease in data:
            labels[disease] = pd.to_numeric(data[disease], errors="coerce")
        else:
            labels[disease] = np.nan
            warnings.append(f"Missing disease column: {disease}")

    patient_ids = data["Path"].map(patient_id) if "Path" in data else pd.Series(np.nan, index=data.index)
    if "Path" not in data:
        warnings.append("No Path column: patient and image-file checks are incomplete.")
    dataset_stats = pd.DataFrame(
        [
            ("Image records analyzed", len(data)),
            ("Unique patients from image paths", patient_ids.nunique()),
            ("Rows with unparseable/missing patient ID", patient_ids.isna().sum()),
            ("SAMPLE_SIZE configuration", "All records" if SAMPLE_SIZE is None else SAMPLE_SIZE),
        ],
        columns=["Metric", "Value"],
    )

    label_summary = pd.DataFrame(
        {
            "Disease": DISEASES,
            "Positive (1)": [labels[d].eq(1).sum() for d in DISEASES],
            "Negative (0)": [labels[d].eq(0).sum() for d in DISEASES],
            "Uncertain (-1)": [labels[d].eq(-1).sum() for d in DISEASES],
            "Missing (NaN)": [labels[d].isna().sum() for d in DISEASES],
        }
    )
    positive = labels.eq(1)
    per_image_positive = positive.sum(axis=1)
    positive_counts = (
        per_image_positive.value_counts().reindex(range(6), fill_value=0).rename_axis("Positive disease count").reset_index(name="Image records")
    )
    # Use int32: full-dataset co-occurrence counts exceed int8's range.
    matrix = positive.astype("int32").T.dot(positive.astype("int32")).astype(int)
    matrix = matrix.loc[DISEASES, DISEASES]
    pair_table = pd.DataFrame(
        [
            {"Disease 1": left, "Disease 2": right, "Co-occurrence count (both positive)": int(matrix.loc[left, right])}
            for left, right in combinations(DISEASES, 2)
        ]
    )
    at_least_two = int((per_image_positive >= 2).sum())
    cooccurrence_summary = pd.DataFrame(
        [
            ("Images with zero positive disease labels", int((per_image_positive == 0).sum())),
            ("Images with at least two positive disease labels", at_least_two),
            ("Positive-label rule", "Only CSV label value 1 is positive; -1 and missing are not positive."),
        ],
        columns=["Metric", "Value"],
    )

    positives = positive.sum()
    negatives = labels.eq(0).sum()
    known = positives + negatives
    imbalance = pd.DataFrame(
        {
            "Disease": DISEASES,
            "Known binary labels": known.reindex(DISEASES).to_numpy(),
            "Positive-to-negative ratio": (positives / negatives.replace(0, np.nan)).reindex(DISEASES).round(4).to_numpy(),
            "Positive percentage among known binary labels": (100 * positives / known.replace(0, np.nan)).reindex(DISEASES).round(2).to_numpy(),
        }
    )
    label_and_imbalance = label_summary.merge(imbalance, on="Disease")
    unexpected = labels.where(~labels.isin([-1, 0, 1])).melt(var_name="Disease", value_name="Unexpected numeric label").dropna()
    unexpected = unexpected.groupby(["Disease", "Unexpected numeric label"]).size().reset_index(name="Count")

    duplicate_paths = data.loc[data["Path"].notna() & data["Path"].duplicated(keep=False)].copy() if "Path" in data else pd.DataFrame(columns=data.columns)
    duplicate_rows = data.loc[data.duplicated(keep=False)].copy()
    image_summary, missing_images = image_file_check(data)
    duplicate_summary = pd.DataFrame(
        [
            ("Duplicate image-path rows", len(duplicate_paths)),
            ("Fully duplicated CSV rows", len(duplicate_rows)),
            ("Unexpected numeric label values", int(unexpected["Count"].sum()) if not unexpected.empty else 0),
        ],
        columns=["Check", "Count"],
    )

    if "Frontal/Lateral" in data:
        views = data["Frontal/Lateral"].astype("string").fillna("Missing/unknown").str.strip().replace("", "Missing/unknown")
        view_order = [view for view in ["Frontal", "Lateral", "Missing/unknown"] if view in set(views)]
        view_order += [view for view in sorted(views.dropna().unique()) if view not in view_order]
        counts = views.value_counts().reindex(view_order, fill_value=0)
        view_counts = counts.rename_axis("View").reset_index(name="Image records")
        view_counts["Percentage"] = (100 * view_counts["Image records"] / len(data)).round(2)
        by_view = positive.assign(View=views).groupby("View")[DISEASES].sum().reindex(view_order, fill_value=0)
        view_labels = by_view.reset_index().melt(id_vars="View", var_name="Disease", value_name="Positive labels")
        view_labels["Images in view"] = view_labels["View"].map(counts)
        view_labels["Positive percentage"] = (100 * view_labels["Positive labels"] / view_labels["Images in view"]).round(2)
    else:
        warnings.append("No Frontal/Lateral column: view analysis is incomplete.")
        view_counts = pd.DataFrame(columns=["View", "Image records", "Percentage"])
        view_labels = pd.DataFrame(columns=["View", "Disease", "Positive labels", "Images in view", "Positive percentage"])

    status = pd.DataFrame(
        [{"Status": "Completed", "Detail": "Metadata-only analysis completed."}]
        + [{"Status": "Warning", "Detail": warning} for warning in warnings]
    )
    overview_rows = []
    for category, table, metric, value, status_col, detail in [
        ("Dataset statistics", dataset_stats, "Metric", "Value", None, None),
        ("Co-occurrence", cooccurrence_summary, "Metric", "Value", None, None),
        ("Data quality", duplicate_summary, "Check", "Count", "Completed", None),
        ("Image-file check", image_summary, "Check", "Value", "Status", "Details"),
    ]:
        for _, row in table.iterrows():
            overview_rows.append(
                {
                    "Category": category,
                    "Metric": row[metric],
                    "Status": row[status_col] if status_col in row.index else (status_col or ""),
                    "Value": row[value],
                    "Details": row[detail] if detail and detail in row.index else "",
                }
            )
    for _, row in status.iterrows():
        overview_rows.append({"Category": "Workflow", "Metric": "Run status", "Status": row["Status"], "Value": "", "Details": row["Detail"]})
    overview = pd.DataFrame(overview_rows)

    write_workbook(
        {
            "Overview & Quality": [("Run summary and quality checks", overview)],
            "Labels & Imbalance": [("Label distribution and class imbalance", label_and_imbalance)],
            "Co-occurrence": [
                ("Positive disease count per image", positive_counts),
                ("Pairwise positive-label co-occurrence", pair_table),
                ("5 x 5 positive-label co-occurrence matrix", matrix.rename_axis("Disease").reset_index()),
            ],
            "Views": [("Frontal/lateral image counts", view_counts), ("Positive-label distribution by view", view_labels)],
            "Quality Details": [
                ("Unexpected numeric label values", unexpected),
                ("Duplicate image-path rows", duplicate_paths),
                ("Fully duplicated CSV rows", duplicate_rows),
                ("Missing image files", missing_images),
            ],
            "Radiological Concepts": [("Disease-to-concept mapping", concept_mapping())],
        }
    )
    save_figures(label_summary, matrix, view_counts, view_labels)
    top_pair = pair_table.sort_values("Co-occurrence count (both positive)", ascending=False).iloc[0] if not pair_table.empty else None
    write_report(dataset_stats, label_summary, positive_counts, imbalance, view_counts, image_summary, status, at_least_two, top_pair)
    print(f"Analysis complete: {OUTPUT_DIR}")
    print(f"Records analyzed: {len(data):,}; warnings: {len(warnings)}")


if __name__ == "__main__":
    main()
