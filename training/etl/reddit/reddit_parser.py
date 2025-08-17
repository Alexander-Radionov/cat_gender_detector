import os
import re
import requests
from pathlib import Path
from typing import List, Optional
import warnings
from tqdm import tqdm

import praw

from definitions import ROOT_DIR


class RedditPostParser:
    """
    Parser that mirrors the public contract of VKPostParser:
      - __init__(login, password, token, max_posts_in_iteration, batch_delay)
      - parse_post(item)
      - get_posts_batch(group_id, posts_to_parse, offset=1)

    Notes:
      - "group_id" is interpreted as subreddit name.
      - Images and texts are saved under ../data/images and ../data/texts respectively.
      - The "token" arg is unused; kept for signature compatibility.
    """

    def __init__(
        self,
        login: str,
        password: str,
        token: Optional[str],
        max_posts_in_iteration: int,
        batch_delay: float,
    ) -> None:
        # Required creds from env; keep signature parity with VK parser
        client_id = os.getenv("REDDIT_CLIENT_ID")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        user_agent = os.getenv("REDDIT_USER_AGENT", "cat-gender-detector/1.0")

        if not all([client_id, client_secret]):
            raise ValueError(
                "Missing Reddit credentials. Ensure REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, and REDDIT_USER_AGENT are set."
            )

        # If username/password provided, use script (password) flow; otherwise use application-only (read-only)
        if login and password:
            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                username=login,
                password=password,
                user_agent=user_agent,
                check_for_async=False,
            )
        else:
            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent,
                check_for_async=False,
            )
            self.reddit.read_only = True

        self.max_posts_in_iteration = max_posts_in_iteration
        self.batch_delay = batch_delay
        self.user_agent = user_agent

        # Ensure target directories exist
        self.data_root = Path(ROOT_DIR).parent / "data"
        self.texts_dir = self.data_root / "texts"
        self.images_dir = self.data_root / "images"
        self.texts_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

    # --- helpers ---
    @staticmethod
    def _clean_url(url: str) -> str:
        # Reddit often returns HTML-escaped urls
        return re.sub(r"&amp;", "&", url)

    def _download_images_from_submission(self, submission) -> List[bytes]:
        image_contents: List[bytes] = []

        # Prefer galleries
        try:
            if getattr(submission, "is_gallery", False) and getattr(submission, "media_metadata", None):
                for _, media in submission.media_metadata.items():
                    # Select the best available (original) if present, else source
                    url = self._clean_url(media.get("s", {}).get("u") or media.get("s", {}).get("gif"))
                    if not url:
                        # fallback to preview sizes if available
                        previews = media.get("p", [])
                        if previews:
                            url = self._clean_url(previews[-1].get("u"))
                    if url:
                        try:
                            resp = requests.get(url, headers={"User-Agent": self.user_agent}, timeout=30)
                            if resp.ok and resp.headers.get("Content-Type", "").startswith("image/"):
                                image_contents.append(resp.content)
                        except Exception:
                            continue
                return image_contents
        except Exception as e:
            warnings.warn(f"Failed to parse gallery images for submission {getattr(submission, 'id', '?')}: {e}")

        # Direct image URL
        if isinstance(submission.url, str) and re.search(r"\.(png|jpe?g|webp)$", submission.url, re.I):
            try:
                resp = requests.get(self._clean_url(submission.url), headers={"User-Agent": self.user_agent}, timeout=30)
                if resp.ok and resp.headers.get("Content-Type", "").startswith("image/"):
                    image_contents.append(resp.content)
            except Exception as e:
                warnings.warn(f"Failed to download direct image for submission {getattr(submission, 'id', '?')}: {e}")

        # Preview fallback
        try:
            preview = getattr(submission, "preview", None)
            if preview and "images" in preview and preview["images"]:
                url = self._clean_url(preview["images"][0]["source"]["url"])  # largest available
                resp = requests.get(url, headers={"User-Agent": self.user_agent}, timeout=30)
                if resp.ok and resp.headers.get("Content-Type", "").startswith("image/"):
                    image_contents.append(resp.content)
        except Exception as e:
            warnings.warn(f"Failed to download preview image for submission {getattr(submission, 'id', '?')}: {e}")

        return image_contents

    # --- required contract ---
    def parse_post(self, submission) -> None:
        post_id = submission.id
        title = getattr(submission, "title", "") or ""
        selftext = getattr(submission, "selftext", "") or ""
        text = (title + "\n\n" + selftext).strip()

        text_path = str(self.texts_dir / f"{post_id}.txt")

        images = self._download_images_from_submission(submission)
        if len(images) == 0:
            warnings.warn(f"Post '{post_id}' has no images. Skipping it")
            return None

        for idx, img_bytes in enumerate(images, start=1):
            img_path = text_path.replace("texts", "images").replace(".txt", f"_{idx}.png")
            with open(img_path, "wb") as f:
                f.write(img_bytes)

        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text)

    def get_posts_batch(self, group_id, posts_to_parse, offset: int = 1) -> None:
        """
        Fetch a batch of posts from a subreddit and parse them.

        Parameters
        - group_id: subreddit name (string), kept for signature compatibility
        - posts_to_parse: number of posts to process in this batch
        - offset: number of newest posts to skip (treated like already parsed)
        """
        if not isinstance(group_id, str):
            group_id = str(group_id)

        subreddit = self.reddit.subreddit(group_id)

        # Announce target storage directories
        print(f"Texts will be stored in: {self.texts_dir}")
        print(f"Images will be stored in: {self.images_dir}")

        # Grab more than we need to honor the offset
        limit = int(posts_to_parse) + int(offset)
        submissions = list(subreddit.new(limit=limit))

        if not submissions:
            print("No items found from subreddit.new().")
            return None

        # Skip the first 'offset' items
        to_process = submissions[offset:offset + posts_to_parse]
        desc = (
            f"r/{group_id} -> texts: {self.texts_dir} | images: {self.images_dir} | "
            f"processing {posts_to_parse} posts"
        )
        for submission in tqdm(to_process, desc=desc):
            self.parse_post(submission)


