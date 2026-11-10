import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY is missing")

if not MODEL_NAME:
    raise ValueError("GEMINI_MODEL is missing")

client = genai.Client(api_key=API_KEY)


def generate_response(prompt: str) -> str:
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")

    response = client.interactions.create(model=MODEL_NAME, input=prompt)

    return response.output_text
