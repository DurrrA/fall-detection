from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
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
    args = parser.parse_args()

    data_root = Path(args.data_root)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data (same ordering as training: sorted class folder names)
    X, y = load_dataset(data_root, image_dir=args.image_dir, img_size=tuple(args.img_size), normalize=False)
    class_names_from_data = sorted([p.name for p in (data_root / args.image_dir).iterdir() if p.is_dir()])
    class_names = load_labels(Path(args.labels), class_names_from_data)

    ds = make_tf_data(X, y, args.batch_size)
    model = tf.keras.models.load_model(args.checkpoint)

    preds = model.predict(ds, verbose=0)
    if preds.ndim == 2 and preds.shape[1] == 1:
        y_pred = (preds.ravel() >= 0.5).astype("int32")
    else:
        y_pred = np.argmax(preds, axis=1).astype("int32")

    acc = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, target_names=class_names, digits=4)
    cm = confusion_matrix(y, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(report)

    (results_dir / "metrics.json").write_text(json.dumps({"accuracy": float(acc)}, indent=2))
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig((results_dir / "confusion_matrix.png").as_posix())
    plt.close(fig)

if __name__ == "__main__":
    main()