import vk_api
import os
import requests
from tqdm import tqdm
from pathlib import Path
import warnings
from definitions import ROOT_DIR
import time


class VKPostParser:
    def __init__(self, login: str, password: str, token: str, max_posts_in_iteration: int, batch_delay: float):
        vk_session = vk_api.VkApi(login, password, token=token)
        self.api = vk_session.get_api()
        self.max_posts_in_iteration = max_posts_in_iteration
        self.batch_delay = batch_delay

    def get_photo_from_attach(self, attach: dict):
        match attach['type']:
            case 'photo':
                return attach['photo']
            case 'album':
                return attach['album']['thumb']
            case _:
                return None

    def parse_post(self, item: dict):
        post_id = item['id']
        text = item['text']
        text_path = str(Path(ROOT_DIR).parent / 'texts' / f'{post_id}.txt')
        images = []
        for i, attach in enumerate(item['attachments']):
            img_dict = self.get_photo_from_attach(attach)
            if img_dict is None:
                continue
            result = requests.get(img_dict['sizes'][-1]['url'])
            images.append(result)
        if len(images) == 0:
            warnings.warn(f"Post '{post_id}' has no images. Skipping it")
            return None

        for i, img in enumerate(images):
            img_path = text_path.replace('text', 'image').replace('.txt', f'_{i + 1}.png')
            with open(img_path, 'wb') as f:
                f.write(img.content)
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text)

    def get_posts_batch(self, group_id, posts_to_parse, offset=1):
        """ VkApi.method позволяет выполнять запросы к API. В этом примере
            используется метод wall.get (https://vk.com/dev/wall.get) с параметром
            count = 1, т.е. мы получаем один последний пост со стены текущего
            пользователя.
        """
        response = self.api.wall.get(owner_id=group_id, count=posts_to_parse, offset=offset)
        if not response['items']:
            print(f"No items found. Response:\n{response}")
            return None
        texts_dir = Path(ROOT_DIR).parent / 'texts'
        images_dir = Path(ROOT_DIR).parent / 'images'
        print(f"Texts will be stored in: {texts_dir}")
        print(f"Images will be stored in: {images_dir}")

        for item in tqdm(response['items'], desc=f"Processing requested {posts_to_parse} posts to texts: {texts_dir} images: {images_dir}..."):
            self.parse_post(item)


if __name__ == "__main__":
    POSTS_TO_PARSE, MAX_POSTS_IN_ITERATION = int(os.getenv('POSTS_TO_PARSE')), int(os.getenv('MAX_POSTS_IN_ITERATION'))
    GROUP_ID, BATCH_DELAY = int(os.getenv('GROUP_ID')), float(os.getenv('BATCH_DELAY'))
    vk_posts_parser = VKPostParser(os.getenv('VK_LOGIN'), os.getenv('VK_PASSWORD'), os.getenv('VK_TOKEN'),
                                   max_posts_in_iteration=MAX_POSTS_IN_ITERATION,
                                   batch_delay=BATCH_DELAY)
    num_iterations = POSTS_TO_PARSE // MAX_POSTS_IN_ITERATION + int(POSTS_TO_PARSE % MAX_POSTS_IN_ITERATION != 0)
    remaining_posts_to_parse = POSTS_TO_PARSE
    parsed_posts = 0
    for it in range(num_iterations):
        current_posts_to_parse = min(remaining_posts_to_parse, MAX_POSTS_IN_ITERATION)
        vk_posts_parser.get_posts_batch(group_id=GROUP_ID, posts_to_parse=current_posts_to_parse,
                                        offset=parsed_posts)
        parsed_posts += current_posts_to_parse
        time.sleep(BATCH_DELAY)
