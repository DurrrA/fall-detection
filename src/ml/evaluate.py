from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_fscore_support, roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, matthews_corrcoef, balanced_accuracy_score, cohen_kappa_score
)
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from .dataset import load_dataset

def load_labels(labels_path: Path | None, class_names_from_data: list[str]) -> list[str]:
    if labels_path and labels_path.exists():
        mapping = json.loads(labels_path.read_text())
        return [mapping[str(i)] for i in sorted(map(int, mapping.keys()))]
    return class_names_from_data

def make_tf_data(X: np.ndarray, y: np.ndarray, batch_size: int) -> tf.data.Dataset:
    return tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="models/best.keras")
    parser.add_argument("--labels", type=str, default="models/labels.json")
    parser.add_argument("--data-root", type=str, default="data/fall_dataset")
    parser.add_argument("--image-dir", type=str, default="images")
    parser.add_argument("--img-size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--results-dir", type=str, default="data/fall_dataset/results")
    parser.add_argument("--positive-class", type=str, default="fall")  # untuk ROC/PR binary
    args = parser.parse_args()

    data_root = Path(args.data_root)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    X, y = load_dataset(data_root, image_dir=args.image_dir, img_size=tuple(args.img_size), normalize=False)
    class_names_from_data = sorted([p.name for p in (data_root / args.image_dir).iterdir() if p.is_dir()])
    class_names = load_labels(Path(args.labels), class_names_from_data)

    ds = make_tf_data(X, y, args.batch_size)
    model = tf.keras.models.load_model(args.checkpoint)

    preds = model.predict(ds, verbose=0)

    # Probabilitas & prediksi
    if preds.ndim == 2 and preds.shape[1] == 1:
        proba = preds.ravel()
        y_pred = (proba >= 0.5).astype("int32")
    else:
        proba = preds
        y_pred = np.argmax(preds, axis=1).astype("int32")

    # Metrik dasar
    cm = confusion_matrix(y, y_pred, labels=range(len(class_names)))
    acc = accuracy_score(y, y_pred)
    bal_acc = balanced_accuracy_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)
    kappa = cohen_kappa_score(y, y_pred)
    prc, rec, f1, sup = precision_recall_fscore_support(
        y, y_pred, labels=range(len(class_names)), zero_division=0
    )
    report_text = classification_report(y, y_pred, target_names=class_names, digits=4)
    report_dict = classification_report(y, y_pred, target_names=class_names, digits=4, output_dict=True)

    device = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"

    metrics = {
        "device": device,
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "mcc": float(mcc),
        "kappa": float(kappa),
        "per_class": [
            {
                "class": class_names[i],
                "precision": float(prc[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(sup[i]),
            }
            for i in range(len(class_names))
        ],
    }

    # Confusion matrix (absolute)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig((results_dir / "confusion_matrix.png").as_posix())
    plt.close(fig)

    # Confusion matrix (normalized per baris)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format=".2f")
    plt.title("Confusion Matrix (Normalized)")
    fig.tight_layout()
    fig.savefig((results_dir / "confusion_matrix_normalized.png").as_posix())
    plt.close(fig)

    # Rinci TP/TN/FP/FN per kelas + sensitivity & specificity
    total = int(cm.sum())
    confusion_details = []
    for i, cname in enumerate(class_names):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(total - tp - fp - fn)
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        confusion_details.append(
            {"class": cname, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
             "sensitivity": sensitivity, "specificity": specificity}
        )
    metrics["confusion_details"] = confusion_details

    # ROC & PR
    if len(class_names) == 2:
        pos_idx = class_names.index(args.positive_class) if args.positive_class in class_names else 1
        y_true_bin = (y == pos_idx).astype(int)
        y_score = proba if proba.ndim == 1 else proba[:, pos_idx]

        fpr, tpr, thr = roc_curve(y_true_bin, y_score)
        prec, recl, pr_thr = precision_recall_curve(y_true_bin, y_score)
        roc_auc = roc_auc_score(y_true_bin, y_score)
        ap = average_precision_score(y_true_bin, y_score)

        j = tpr - fpr
        best_idx = int(np.argmax(j))
        best_thr = float(thr[best_idx]) if best_idx < len(thr) else 0.5

        metrics.update({
            "roc_auc": float(roc_auc),
            "avg_precision": float(ap),
            "best_threshold": best_thr,
        })

        # Plot ROC
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curve"); ax.legend()
        fig.tight_layout()
        fig.savefig((results_dir / "roc_curve.png").as_posix())
        plt.close(fig)

        # Plot PR
        fig, ax = plt.subplots()
        ax.plot(recl, prec, label=f"AP={ap:.3f}")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-Recall Curve"); ax.legend()
        fig.tight_layout()
        fig.savefig((results_dir / "pr_curve.png").as_posix())
        plt.close(fig)
    else:
        # One-vs-rest untuk multi-class
        Y_bin = label_binarize(y, classes=range(len(class_names)))
        roc_aucs = {}
        aps = {}
        for i, cname in enumerate(class_names):
            y_true_i = Y_bin[:, i]
            y_score_i = proba[:, i]
            fpr, tpr, _ = roc_curve(y_true_i, y_score_i)
            prec, recl, _ = precision_recall_curve(y_true_i, y_score_i)
            roc_auc = roc_auc_score(y_true_i, y_score_i)
            ap = average_precision_score(y_true_i, y_score_i)
            roc_aucs[cname] = float(roc_auc)
            aps[cname] = float(ap)

            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title(f"ROC Curve - {cname}"); ax.legend()
            fig.tight_layout()
            fig.savefig((results_dir / f"roc_curve_{cname}.png").as_posix())
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.plot(recl, prec, label=f"AP={ap:.3f}")
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title(f"PR Curve - {cname}"); ax.legend()
            fig.tight_layout()
            fig.savefig((results_dir / f"pr_curve_{cname}.png").as_posix())
            plt.close(fig)
        metrics.update({"roc_auc_ovr": roc_aucs, "avg_precision_ovr": aps})

    # Tulis artefak
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (results_dir / "classification_report.json").write_text(json.dumps(report_dict, indent=2))

    # Cetak ringkas ke terminal
    print(f"Accuracy: {acc:.4f}  Balanced: {bal_acc:.4f}  MCC: {mcc:.4f}  Kappa: {kappa:.4f}")
    print(report_text)

if __name__ == "__main__":
    main()