import os
import re
import io
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

from tqdm import tqdm
from telethon import TelegramClient
from telethon.tl.types import Message

from definitions import ROOT_DIR


class TelegramPostParser:
    """
    Parser that mirrors the public contract of VKPostParser/RedditPostParser:
      - __init__(login, password, token, max_posts_in_iteration, batch_delay)
      - parse_post(item)
      - get_posts_batch(group_id, posts_to_parse, offset=1)

    Notes:
      - "group_id" is interpreted as channel username or URL (e.g., "murkosha", "@murkosha", or "https://t.me/murkosha").
      - Images and texts are saved under ../data/images and ../data/texts respectively.
      - login/password/token args are unused; kept for signature compatibility.
      - Requires environment variables TELEGRAM_API_ID and TELEGRAM_API_HASH.
        Optionally TELEGRAM_SESSION (path to session file, defaults to ".telegram_session").
      - Posts that contain any video are skipped entirely.
    """

    def __init__(
        self,
        login: str,
        password: str,
        token: Optional[str],
        max_posts_in_iteration: int,
        batch_delay: float,
    ) -> None:
        api_id = os.getenv("TELEGRAM_API_ID")
        api_hash = os.getenv("TELEGRAM_API_HASH")
        session_path = os.getenv("TELEGRAM_SESSION", ".telegram_session")

        if not api_id or not api_hash:
            raise ValueError(
                "Missing Telegram credentials. Ensure TELEGRAM_API_ID and TELEGRAM_API_HASH are set."
            )

        # Create client; assumes session already authorized (first run may require interactive login)
        # We intentionally do not attempt interactive code requests here.
        self.client = TelegramClient(session=session_path, api_id=int(api_id), api_hash=api_hash)

        self.max_posts_in_iteration = max_posts_in_iteration
        self.batch_delay = batch_delay

        # Ensure target directories exist
        self.data_root = Path(ROOT_DIR).parent / "data"
        self.texts_dir = self.data_root / "texts"
        self.images_dir = self.data_root / "images"
        self.texts_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

    # --- helpers ---
    @staticmethod
    def _normalize_channel(group_id: str) -> str:
        if not isinstance(group_id, str):
            group_id = str(group_id)
        group_id = group_id.strip()
        # Accept forms like t.me/xxx, https://t.me/xxx, @xxx, or plain xxx
        group_id = re.sub(r"^https?://t\.me/", "", group_id, flags=re.IGNORECASE)
        group_id = re.sub(r"^t\.me/", "", group_id, flags=re.IGNORECASE)
        group_id = re.sub(r"^@", "", group_id)
        return group_id

    @staticmethod
    def _message_has_video(msg: Message) -> bool:
        try:
            if getattr(msg, "video", None):
                return True
            doc = getattr(msg, "document", None)
            if doc and getattr(doc, "mime_type", "").startswith("video"):
                return True
            # Video notes / round videos
            if getattr(msg, "video_note", None):
                return True
        except Exception:
            return False
        return False

    @staticmethod
    def _message_has_photo(msg: Message) -> bool:
        try:
            if getattr(msg, "photo", None):
                return True
            # Some photos can be documents with image mimetype
            doc = getattr(msg, "document", None)
            if doc and getattr(doc, "mime_type", "").startswith("image"):
                return True
        except Exception:
            return False
        return False

    def _download_photo_bytes(self, msg: Message) -> Optional[bytes]:
        try:
            buffer = io.BytesIO()
            self.client.loop.run_until_complete(self.client.download_media(msg, file=buffer))
            return buffer.getvalue()
        except Exception:
            return None

    # --- required contract ---
    def parse_post(self, item: Tuple[int, List[Message], str]) -> None:
        """
        item is a tuple: (post_id, messages_in_group, combined_text)
        """
        post_id, messages, text = item
        text_path = str(self.texts_dir / f"{post_id}.txt")

        # Collect image bytes from all messages in the group
        image_bytes_list: List[bytes] = []
        for msg in messages:
            if self._message_has_photo(msg):
                img_bytes = self._download_photo_bytes(msg)
                if img_bytes:
                    image_bytes_list.append(img_bytes)

        if len(image_bytes_list) == 0:
            warnings.warn(f"Post '{post_id}' has no images. Skipping it")
            return None

        for idx, img_bytes in enumerate(image_bytes_list, start=1):
            img_path = text_path.replace("texts", "images").replace(".txt", f"_{idx}.png")
            with open(img_path, "wb") as f:
                f.write(img_bytes)

        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text)

    async def _fetch_messages(self, channel: str, limit: int) -> List[Message]:
        messages: List[Message] = []
        async with self.client:
            async for msg in self.client.iter_messages(entity=channel, limit=limit):
                messages.append(msg)
        return messages

    def _group_messages_into_posts(self, messages: List[Message]) -> List[Tuple[int, List[Message], str]]:
        """
        Groups messages by album (grouped_id). For single messages, each is its own post.
        Each grouped post is represented as (post_id, [messages], combined_text).
        - Skip any group that contains a video.
        - Combined text is caption/text from messages concatenated.
        - post_id is the smallest message.id in the group (stable for album).
        """
        groups: dict[Optional[int], List[Message]] = {}
        singles: List[Message] = []
        for msg in messages:
            gid = getattr(msg, "grouped_id", None)
            if gid:
                groups.setdefault(gid, []).append(msg)
            else:
                singles.append(msg)

        posts: List[Tuple[int, List[Message], str]] = []

        # Album groups
        for gid, msgs in groups.items():
            if any(self._message_has_video(m) for m in msgs):
                # Skip posts with videos entirely
                continue
            # Must contain at least one photo
            if not any(self._message_has_photo(m) for m in msgs):
                continue
            post_id = min(m.id for m in msgs)
            combined_text = "\n\n".join([m.message for m in msgs if (m.message or "").strip()])
            posts.append((post_id, msgs, combined_text))

        # Single messages
        for m in singles:
            if self._message_has_video(m):
                continue
            if not self._message_has_photo(m):
                continue
            post_id = m.id
            text = (m.message or "").strip()
            posts.append((post_id, [m], text))

        # Keep original order (newest to oldest) based on post_id descending
        posts.sort(key=lambda p: p[0], reverse=True)
        return posts

    def get_posts_batch(self, group_id, posts_to_parse, offset: int = 1) -> None:
        channel = self._normalize_channel(group_id)

        # Announce target storage directories
        print(f"Texts will be stored in: {self.texts_dir}")
        print(f"Images will be stored in: {self.images_dir}")

        # We fetch more raw messages to honor offset and skipping rules
        # Start with a generous limit; adjust if needed
        raw_limit = int(posts_to_parse) + int(offset) + int(self.max_posts_in_iteration) * 2

        # Fetch messages (newest first)
        messages = self.client.loop.run_until_complete(self._fetch_messages(channel=channel, limit=raw_limit))
        if not messages:
            print("No items found from the channel.")
            return None

        # Group into logical posts and apply skipping rules
        posts = self._group_messages_into_posts(messages)
        if not posts:
            print("No suitable posts (with photos and without videos) found.")
            return None

        # Respect offset, then process the requested count
        to_process = posts[offset:offset + posts_to_parse]
        desc = (
            f"t.me/{channel} -> texts: {self.texts_dir} | images: {self.images_dir} | "
            f"processing {len(to_process)} posts"
        )
        for post in tqdm(to_process, desc=desc):
            self.parse_post(post)



