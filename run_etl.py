import os
import dotenv
from pathlib import Path

from definitions import (ROOT_DIR, MODEL_PATH, GROUP_ID, POSTS_TO_PARSE, MAX_POSTS_IN_ITERATION, BATCH_DELAY,
                         LLM_PROVIDER, LLM_TEMPERATURE, LLM_NAME)
from training.etl.etl import ETL
from training.etl.training_description_classifier import DescriptionClassifier
from training.etl.training_image_processor import TrainingImageProcessor
from training.etl.dataset_manager import DatasetManager
from training.etl.reddit.reddit_parser import RedditPostParser
from training.etl.telegram import TelegramPostParser

# Load environment variables
dotenv.load_dotenv()


def run_etl():
    # Initialize components
    text_language = os.getenv('TEXT_LANGUAGE', 'ru')
    print(f"Text language: {text_language}")
    description_classifier = DescriptionClassifier(provider=LLM_PROVIDER, model_name=LLM_NAME,
                                                   temperature=LLM_TEMPERATURE, language=text_language)
    train_image_processor = TrainingImageProcessor(model_path=MODEL_PATH)
    dataset_manager = DatasetManager()

    # Select source via env: SOURCE in {"reddit", "telegram"}; default reddit
    source = (os.getenv('SOURCE') or 'reddit').lower().strip()
    reddit_login = os.getenv('REDDIT_USERNAME')
    reddit_password = os.getenv('REDDIT_PASSWORD')

    if source == 'telegram':
        post_parser = TelegramPostParser(
            login='',
            password='',
            token=None,
            max_posts_in_iteration=int(MAX_POSTS_IN_ITERATION),
            batch_delay=float(BATCH_DELAY)
        )
    else:
        post_parser = RedditPostParser(
            login=reddit_login,
            password=reddit_password,
            token=None,
            max_posts_in_iteration=int(MAX_POSTS_IN_ITERATION),
            batch_delay=float(BATCH_DELAY)
        )

    etl = ETL(
        description_classifier=description_classifier,
        train_image_processor=train_image_processor,
        dataset_manager=dataset_manager,
        post_parser=post_parser,
    )
    
    # Run the ETL pipeline
    if source == 'telegram':
        # Default to murkosha channel if none provided
        channel = os.getenv('TELEGRAM_CHANNEL') or 'murkosha'
        group_id_val = channel
    else:
        # Interpret GROUP_ID as subreddit name for backwards compatibility
        # Prefer REDDIT_SUBREDDIT if set
        subreddit = os.getenv('REDDIT_SUBREDDIT') or str(GROUP_ID)
        group_id_val = subreddit
    posts_to_parse = int(POSTS_TO_PARSE)
    max_posts_in_iteration = int(MAX_POSTS_IN_ITERATION)
    batch_delay = float(BATCH_DELAY)
    
    stats = etl.run_etl_pipeline(
        group_id=group_id_val,
        posts_to_parse=posts_to_parse,
        max_posts_in_iteration=max_posts_in_iteration,
        batch_delay=batch_delay
    )
    
    print("ETL pipeline completed successfully!")
    print(f"Dataset statistics: {stats}")


if __name__ == "__main__":
    run_etl()
