import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

def prepare_data(data_csv_path, images_dir, output_dir, test_size=0.2):
    # Load the dataset
    data = pd.read_csv(data_csv_path)

    # Encode labels
    label_encoder = LabelEncoder()
    data['label'] = label_encoder.fit_transform(data['label'])

    # Split the dataset into training and validation sets
    train_data, val_data = train_test_split(data, test_size=test_size, random_state=42)

    # Create output directories
    train_output_dir = Path(output_dir) / 'train'
    val_output_dir = Path(output_dir) / 'val'
    train_output_dir.mkdir(parents=True, exist_ok=True)
    val_output_dir.mkdir(parents=True, exist_ok=True)

    # Copy images to the respective directories
    for index, row in train_data.iterrows():
        img_path = Path(images_dir) / row['filename']
        if img_path.exists():
            img_dest = train_output_dir / row['filename']
            img_dest.write_text(img_path.read_text())

    for index, row in val_data.iterrows():
        img_path = Path(images_dir) / row['filename']
        if img_path.exists():
            img_dest = val_output_dir / row['filename']
            img_dest.write_text(img_path.read_text())

    # Save the processed data
    train_data.to_csv(train_output_dir / 'data.csv', index=False)
    val_data.to_csv(val_output_dir / 'data.csv', index=False)

if __name__ == "__main__":
    prepare_data(
        data_csv_path='data/fall_dataset/data.csv',
        images_dir='data/fall_dataset/images',
        output_dir='data/fall_dataset/processed'
    )