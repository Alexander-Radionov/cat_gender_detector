import os
import re
import json
import time
import pandas as pd
from glob import glob
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

from definitions import ROOT_DIR, EXCEPTIONS_FILE_NAME, TRAIN_IMAGE_FORMAT
from training.etl.training_description_classifier import DescriptionClassifier
from training.etl.training_image_processor import TrainingImageProcessor
from training.etl.dataset_manager import DatasetManager


class ETL:
    def __init__(self, description_classifier: DescriptionClassifier, 
                 train_image_processor: TrainingImageProcessor,
                 dataset_manager: DatasetManager = None,
                 post_parser: object | None = None):
        self.description_classifier = description_classifier
        self.train_image_processor = train_image_processor
        self.dataset_manager = dataset_manager or DatasetManager(image_format=TRAIN_IMAGE_FORMAT)
        self.post_parser = post_parser

        self.classes_log_name = "image_processing_log.csv"
        if os.path.exists(self.classes_log_name) and os.path.getsize(self.classes_log_name) > 0:
            try:
                log_df = pd.read_csv(self.classes_log_name)
                self.classes_log = log_df.to_dict(orient='records')
            except Exception:
                # Corrupt or unreadable file: start fresh
                self.classes_log = []
        else:
            self.classes_log = []

        self.exceptions_file_name = EXCEPTIONS_FILE_NAME

        # Ensure data directories exist
        data_path = Path(ROOT_DIR).parent / 'data'
        for subdir in ['texts', 'images', 'labels']:
            os.makedirs(data_path / subdir, exist_ok=True)

    def image_path_to_text_path(self, image_path):
        text_path = image_path.replace('images', 'texts').replace(self.dataset_manager.image_format, '.txt')
        text_path = re.sub(r'_\d+', '', text_path)
        return text_path

    def get_correct_class_name(self, text_path: str):
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read()
            correct_class_name = self.description_classifier.classify_description(text)
            print(f"Correct class name: {correct_class_name}")

            # Log the classification
            self.classes_log.append({
                'text_path': text_path,
                'class_name': correct_class_name,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            })

            return correct_class_name
        except Exception as e:
            if not os.path.exists(self.exceptions_file_name):
                os.makedirs(os.path.dirname(self.exceptions_file_name), exist_ok=True)
            with open(self.exceptions_file_name, 'a') as f:
                f.write(f"Text path: {text_path}. Error: {str(e)}\n")
            return "OTHER"  # Default to OTHER in case of error

    def process_training_post(self, text_file_path: str, processed_text_file_names: list[str],
                              images_path: str) -> tuple[bool, str]:
        if Path(text_file_path).name in processed_text_file_names:
            return False, "processed"

        description_class_name = self.get_correct_class_name(text_file_path)
        description_class_id = self.description_classifier.get_class_id(description_class_name)

        # 2 means no cat is on the image or there are multiple cats
        if description_class_id != 2:
            image_file_search_pattern = f"{images_path}/{Path(text_file_path).stem}*{self.dataset_manager.image_format}"
            image_paths = glob(image_file_search_pattern)

            if not image_paths:
                return False, "No images found for this text"

            processed_images = 0
            for img_path in image_paths:
                success = self.train_image_processor.process_training_image(img_path, description_class_id)
                processed_images += int(success)

            if processed_images == 0:
                return False, "Text classified, but no images were processed"
            return True, "OK"
        else:
            return False, "No cats mentioned or multiple cats mentioned"

    def get_processed_text_file_names(self, labels_path: str):
        label_files = glob(f"{labels_path}/*.txt")
        processed_text_files = []

        for l_file in label_files:
            text_file = self.image_path_to_text_path(l_file)
            text_file_name = Path(text_file).name
            processed_text_files.append(text_file_name)

        return processed_text_files

    def parse_posts(self, group_id, posts_to_parse, max_posts_in_iteration=100, batch_delay=1.0) -> int:
        """Parse posts from a source and save them to the data directory using the configured post_parser."""
        if not self.post_parser:
            raise ValueError("Post parser is not initialized")

        print(f"Starting to parse {posts_to_parse} posts from source {group_id}")

        num_iterations = posts_to_parse // max_posts_in_iteration + int(posts_to_parse % max_posts_in_iteration != 0)
        remaining_posts_to_parse = posts_to_parse
        parsed_posts = 0

        for it in range(num_iterations):
            current_posts_to_parse = min(remaining_posts_to_parse, max_posts_in_iteration)
            self.post_parser.get_posts_batch(
                group_id=group_id,
                posts_to_parse=current_posts_to_parse,
                offset=parsed_posts,
            )
            parsed_posts += current_posts_to_parse
            remaining_posts_to_parse -= current_posts_to_parse

            if it < num_iterations - 1:  # Don't sleep after the last iteration
                time.sleep(batch_delay)

        print(f"Finished parsing {parsed_posts} posts from source")
        return parsed_posts

    def process_training_data(self, sleep_time=0.5):
        """Process all text files and create training items"""
        data_path = Path(ROOT_DIR).parent / 'data'
        texts_path = str(data_path / 'texts')
        images_path = str(data_path / 'images')
        labels_path = str(data_path / 'labels')

        text_file_paths = glob(f"{texts_path}/*.txt")
        processed_text_file_names = self.get_processed_text_file_names(labels_path)
        skipped_text_files = defaultdict(list)

        print(f"Found {len(text_file_paths)} text files to process")
        print(f"Already processed {len(processed_text_file_names)} text files")
        
        for t_path in tqdm(text_file_paths, desc="Processing text files"):
            success, reason = self.process_training_post(t_path, processed_text_file_names, images_path)
            if not success:
                skipped_text_files[reason].append(t_path)
                continue
            
            if sleep_time > 0:
                time.sleep(sleep_time)  # To avoid rate limiting with OpenAI API

        print(f"\nFinished processing text files.")
        for reason, files in skipped_text_files.items():
            print(f"Skipped {len(files)} files due to: {reason}")
        
        with open('texts_skipped_log.json', 'w') as f:
            json.dump(skipped_text_files, f, indent=4)
        
        log_df = pd.DataFrame.from_records(self.classes_log)
        log_df.to_csv(self.classes_log_name, index=False)

        stats = self.dataset_manager.organize_dataset()
        print(f"Dataset organized: {stats['train']} training, {stats['val']} validation, {stats['test']} test images")

        yaml_path = self.dataset_manager.generate_data_yaml()
        print(f"Generated data.yaml at {yaml_path}")

        return stats
    
    def run_etl_pipeline(self, group_id, posts_to_parse, max_posts_in_iteration=100, batch_delay=1.0, sleep_time=0.5):
        """Run the complete ETL pipeline from parsing to dataset creation"""
        if self.post_parser:
            parsed_posts = self.parse_posts(
                group_id=group_id,
                posts_to_parse=posts_to_parse,
                max_posts_in_iteration=max_posts_in_iteration,
                batch_delay=batch_delay,
            )
            print(f"Successfully parsed {parsed_posts} posts from source")
        else:
            print("Post parser not initialized, skipping post parsing step")

        stats = self.process_training_data(sleep_time=sleep_time)
        # Ensure telemetry is flushed before exit in short-lived runs
        try:
            if hasattr(self.description_classifier, "flush_traces"):
                self.description_classifier.flush_traces()
        except Exception:
            pass
        
        return stats
