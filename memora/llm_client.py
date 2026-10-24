import os
import time
from dataclasses import dataclass
from urllib import response

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    latency_ms: float

class GroqLLMClient:
    def __init__(self) -> None:
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError("GROQ_API_KEY is missing from .env")

        self.model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

        self.client = Groq(
            api_key=api_key,
            timeout=30.0,
            max_retries=2
        )

    def generate(self, prompt: str, system_prompt: str | None=None, max_completion_tokens: int=512) -> LLMResponse:
                if not prompt.strip():
                    raise ValueError("Prompt cannot be empty.")
    
                if max_completion_tokens <= 0:
                    raise ValueError("max_completion_tokens must be a positive integer.")
    
                messages = []
    
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
    
                messages.append({"role": "user", "content": prompt})
    
                start_time = time.perf_counter()
    
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=max_completion_tokens
                )
    
                latency_ms = (time.perf_counter() - start_time) * 1000
    
                choice = response.choices[0]
                usage = response.usage
    
                return LLMResponse(
                    text=choice.message.content or "",
                    model=response.model,
                    prompt_tokens=usage.prompt_tokens if usage else None,
                    completion_tokens=usage.completion_tokens if usage else None,
                    total_tokens=usage.total_tokens if usage else None,
                    latency_ms=latency_ms
                )
    
    
    