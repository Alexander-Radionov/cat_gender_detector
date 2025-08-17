from typing import Literal
from openai import OpenAI
import dotenv
import os
dotenv.load_dotenv()
from prompts import get_text_classifier_prompt
from langfuse import Langfuse
from langfuse.openai import OpenAI as LangfuseOpenAI


class DescriptionClassifier:
    def __init__(self, provider: Literal["openai", "deepseek"], model_name, temperature=0.01, language: str = "ru"):
        self.langfuse = Langfuse()
        print(f"Provider: {provider}")
        match provider:
            case "openai":
                # Wrap OpenAI client so requests/usage are auto-tracked by Langfuse
                self.client = LangfuseOpenAI()
            case "deepseek":
                self.client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
            case _:
                raise ValueError(f"\nLLM provider has to be openai or deepseek. Got '{provider}' instead")
        self.model_name = model_name
        self.temperature = temperature
        self.language = language
        self.class_names_mapping = {"MALE CAT": 0, "FEMALE CAT": 1, "OTHER": 2}
        self.provider = provider

    def get_class_id(self, class_name_candidate) -> int:
        if class_name_candidate not in self.class_names_mapping:
            actual_class_names = [name for name in self.class_names_mapping.keys() if name in class_name_candidate]
            if len(actual_class_names) > 1:
                return self.class_names_mapping['OTHER']
            else:
                return self.class_names_mapping[actual_class_names[0]]
        return self.class_names_mapping[class_name_candidate]

    def classify_description(self, text: str):
        system_prompt = get_text_classifier_prompt(self.language)
        # Use Langfuse OpenAI wrapper (auto-instrumentation). Also create a root span.
        with self.langfuse.start_as_current_span(name="text_classification") as span:
            # OpenTelemetry spans expose set_attribute(name, value)
            try:
                span.set_attribute("language", self.language)
                span.set_attribute("provider", self.provider)
                span.set_attribute("model", self.model_name)
            except Exception:
                pass
            completion = self.client.chat.completions.create(
                model=self.model_name,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
            )
            result = completion.choices[0].message.content
            try:
                span.set_attribute("result", result)
            except Exception:
                pass
            return result

    def flush_traces(self):
        try:
            self.langfuse.flush()
        except Exception:
            pass
