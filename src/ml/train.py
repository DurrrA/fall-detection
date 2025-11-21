from __future__ import annotations
import argparse
import json
from pathlib import Path
import os, random, numpy as np, tensorflow as tf
from sklearn.model_selection import train_test_split

from .dataset import load_dataset
from .model import create_model

AUTOTUNE = tf.data.AUTOTUNE
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED); os.environ["PYTHONHASHSEED"]=str(SEED)

def get_class_names(dataset_root: Path, image_dir: str) -> list[str]:
    data_dir = (dataset_root / image_dir)
    return sorted([d.name for d in data_dir.iterdir() if d.is_dir()])

def make_tf_data(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    augment_model: tf.keras.Model | None = None,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(X), 1000), reshuffle_each_iteration=True)
    if augment_model is not None:
        ds = ds.map(lambda img, lbl: (augment_model(img, training=True), lbl), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=str, default="images")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--img-size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--weights", type=str, default="imagenet", help="imagenet or none")
    parser.add_argument("--train-base", action="store_true", help="Unfreeze base (fine-tune)")
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation")
    args = parser.parse_args()

    dataset_root = Path("data/fall_dataset")
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    class_names = get_class_names(dataset_root, args.image_dir)
    if len(class_names) < 2:
        raise RuntimeError(f"Expected at least 2 classes in {dataset_root/args.image_dir}, found {class_names}")
    print(f"Classes: {class_names}")

    # Load and split
    X, y = load_dataset(dataset_root, image_dir=args.image_dir, img_size=tuple(args.img_size), normalize=False)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=args.test_split, random_state=SEED, stratify=y
    )
    val_ratio = args.val_split / (1.0 - args.test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=SEED, stratify=y_temp
    )
    print(f"Split -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    # Class weights to counter imbalance
    uniq, cnts = np.unique(y_train, return_counts=True)
    total = y_train.shape[0]
    class_weight = {int(k): float(total / (len(uniq) * c)) for k, c in zip(uniq, cnts)}
    print(f"class_weight: {class_weight}")

    # Augmentation
    augment = None
    if not args.no_augment:
        augment = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.05),
                tf.keras.layers.RandomZoom(0.1),
                tf.keras.layers.RandomContrast(0.1),
            ],
            name="augment",
        )

    # Model
    model = create_model(
        num_classes=len(class_names),
        input_shape=(args.img_size[0], args.img_size[1], 3),
        learning_rate=args.learning_rate,
        metrics=["accuracy"],
        weights=None if args.weights.lower() == "none" else args.weights,
        train_base=bool(args.train_base),
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=(models_dir / "best.keras").as_posix(),
            monitor="val_accuracy",
            save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]

    train_ds = make_tf_data(X_train, y_train, args.batch_size, shuffle=True, augment_model=augment)
    val_ds = make_tf_data(X_val, y_val, args.batch_size, shuffle=False, augment_model=None)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    labels_json = {str(i): name for i, name in enumerate(class_names)}
    (models_dir / "labels.json").write_text(json.dumps(labels_json, indent=2))
    model.save((models_dir / "last.keras").as_posix())

    print("Training done.")
    print(f"Best model: {models_dir / 'best.keras'}")
    print(f"Labels: {models_dir / 'labels.json'}")

if __name__ == "__main__":
    main()