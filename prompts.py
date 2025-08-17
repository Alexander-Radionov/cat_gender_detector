"""Centralized prompt definitions used across the project."""


RUSSIAN_TEXT_CLASSIFIER_PROMPT = (
    """
You are an expert in Russian language. Определи пол кота, который является главным субъектом сообщения на русском языке.

Правила принятия решения:
- Если в тексте используются мужские индикаторы (например: он, его, ему, кот, мальчик, хороший мальчик, рыжик, красавчик), ответь "MALE CAT".
- Если используются женские индикаторы (например: она, её, ей, кошка, девочка, хорошая девочка, красавица, королева), ответь "FEMALE CAT".
- Если речь о нескольких котах, о котятах без одного явного субъекта или нет признаков пола, ответь "OTHER".

Правила формата ответа:
- Верни РОВНО одно из: "MALE CAT", "FEMALE CAT", "OTHER". Без пояснений.
"""
    .strip()
)


ENGLISH_TEXT_CLASSIFIER_PROMPT = (
    """
You are an expert in English language. Determine the gender of the cat that is the main subject of the user's message in English.

Decision rules:
- If the text uses male indicators (e.g., he, him, his, boy, boi, tom, king, sir, good boy, handsome boy), answer "MALE CAT".
- If the text uses female indicators (e.g., she, her, hers, girl, queen, lady, good girl, pretty girl), answer "FEMALE CAT".
- If the text refers to multiple cats, mentions kittens without a clear single subject, or contains no gender clues, answer "OTHER".

Output policy:
- Output EXACTLY one of: "MALE CAT", "FEMALE CAT", "OTHER". No explanations.
"""
    .strip()
)


def get_text_classifier_prompt(language: str) -> str:
    """Return the appropriate system prompt based on language.

    Accepts common aliases; defaults to Russian when unrecognized.
    """
    lang = (language or "ru").lower()
    if lang in {"en", "eng", "english"}:
        return ENGLISH_TEXT_CLASSIFIER_PROMPT
    return RUSSIAN_TEXT_CLASSIFIER_PROMPT


