# Fall Detection Project

This project implements a fall detection system using machine learning techniques. It includes a graphical user interface (GUI) for real-time testing and notifications when a fall is detected.

## Project Structure

```
fall-detection-project
├── src
│   ├── app.py                # Main entry point of the application
│   ├── config
│   │   ├── __init__.py
│   │   └── settings.py       # Configuration settings for the application
│   ├── gui
│   │   ├── __init__.py
│   │   └── main_window.py     # GUI layout and functionality
│   ├── ml
│   │   ├── __init__.py
│   │   ├── dataset.py         # Dataset loading and preprocessing
│   │   ├── model.py           # Machine learning model architecture
│   │   ├── train.py           # Model training logic
│   │   └── evaluate.py        # Model evaluation metrics
│   ├── realtime
│   │   ├── __init__.py
│   │   └── video_stream.py     # Video capture and processing for fall detection
│   ├── notifications
│   │   ├── __init__.py
│   │   ├── notifier.py         # Notification management
│   │   ├── email_provider.py    # Email notification functionality
│   │   └── sms_provider.py      # SMS notification functionality
│   └── utils
│       ├── __init__.py
│       ├── preprocessing.py     # Utility functions for preprocessing
│       └── visualization.py      # Visualization of detection results
├── data
│   └── fall_dataset            # Dataset used for training and testing
├── models
│   └── .gitkeep                # Keeps the models directory in version control
├── tests
│   ├── test_dataset.py         # Unit tests for dataset functions
│   └── test_inference.py       # Unit tests for model inference
├── scripts
│   ├── prepare_data.py         # Prepares dataset for training
│   ├── export_model.py         # Exports trained model for deployment
│   └── run_realtime.py         # Runs the real-time fall detection application
├── requirements.txt            # Project dependencies
├── pyproject.toml              # Project metadata and configuration
├── .env.example                 # Example of environment variables
└── README.md                   # Documentation for the project
```

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd fall-detection-project
   ```

2. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

3. **Prepare the dataset**:
   Run the following script to prepare the dataset for training:
   ```
   python scripts/prepare_data.py
   ```

4. **Train the model**:
   To train the model, execute:
   ```
   python src/ml/train.py
   ```

5. **Run the application**:
   Start the real-time fall detection application with:
   ```
   python scripts/run_realtime.py
   ```

## Usage Guidelines

- The application will capture video from your webcam and process each frame to detect falls.
- Notifications will be sent via email or SMS when a fall is detected, based on the configuration settings.
- You can customize the notification settings in `src/config/settings.py`.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.