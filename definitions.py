import sys
from pathlib import Path

# Root directory of the project (folder that contains this file)
ROOT_DIR = str(Path(__file__).parent)
if not 'cat_gender_detector' in ROOT_DIR:
    ROOT_DIR = str(Path(ROOT_DIR) / 'cat_gender_detector')
EXCEPTIONS_FILE_NAME = str(Path(ROOT_DIR) / "object_detection" / "image_processing_exceptions.log")
MODEL_PATH = str(Path(ROOT_DIR) / "yolov10b.pt")
TRAIN_IMAGE_FORMAT = '.png'
POSTS_TO_PARSE = 2000
MAX_POSTS_IN_ITERATION = 100
GROUP_ID = -10572734
BATCH_DELAY = 4
LLM_PROVIDER = 'deepseek'  # could be OPENAI
LLM_TEMPERATURE = 0.1
LLM_NAME = "deepseek-chat"  # or OPENAI model
