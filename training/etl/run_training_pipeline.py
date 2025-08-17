import os
import dotenv
from pathlib import Path
from definitions import ROOT_DIR, MODEL_PATH
from ultralytics import YOLO


def run_training_pipeline():
    # Load environment variables
    dotenv.load_dotenv(str(Path(ROOT_DIR) / '.env'))

    # Set up training parameters
    data_yaml_path = str(Path(ROOT_DIR).parent / 'data' / 'data.yaml')
    epochs = int(os.getenv('TRAINING_EPOCHS', '100'))
    batch_size = int(os.getenv('BATCH_SIZE', '16'))
    img_size = int(os.getenv('IMAGE_SIZE', '640'))

    # Initialize model
    model = YOLO(MODEL_PATH)

    # Train the model
    print(f"Starting training with {data_yaml_path}...")
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        patience=int(os.getenv('PATIENCE', '20')),
        save=True,
        device=os.getenv('DEVICE', '0')
    )

    # Save the model to inference directory
    inference_dir = Path(ROOT_DIR) / 'inference' / 'models'
    os.makedirs(str(inference_dir), exist_ok=True)
    
    model_path = str(inference_dir / 'cat_gender_detector.pt')
    model.export(format='onnx')  # Export to ONNX format for deployment
    
    print(f"Training completed! Model saved to {model_path}")
    return model_path


if __name__ == "__main__":
    run_training_pipeline()
