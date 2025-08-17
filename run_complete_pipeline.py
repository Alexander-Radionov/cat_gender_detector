import os
import time
from pathlib import Path
from definitions import ROOT_DIR

from training.etl.run_training_pipeline import run_etl_pipeline
from training.run_training_pipeline import run_training_pipeline

def run_complete_pipeline():
    start_time = time.time()
    
    # Step 1: Run ETL pipeline
    print("=" * 50)
    print("STEP 1: Running ETL Pipeline")
    print("=" * 50)
    etl_stats = run_etl_pipeline()
    
    # Step 2: Run training pipeline
    print("\n" + "=" * 50)
    print("STEP 2: Running Training Pipeline")
    print("=" * 50)
    model_path = run_training_pipeline()
    
    # Print summary
    total_time = time.time() - start_time
    print("\n" + "=" * 50)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 50)
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Dataset statistics: {etl_stats}")
    print(f"Trained model saved to: {model_path}")

if __name__ == "__main__":
    run_complete_pipeline() 