import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from app.schemas.research import ResearchPlan

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY is missing")

if not MODEL_NAME:
    raise ValueError("GEMINI_MODEL is missing")


client = genai.Client(
    api_key=API_KEY,
    http_options=types.HttpOptions(
        timeout=30_000,
        retry_options=types.HttpRetryOptions(
            attempts=3,
            initial_delay=1,
            max_delay=8,
            exp_base=2,
            http_status_codes=[408, 429, 500, 502, 503, 504],
        ),
    ),
)


def generate_response(prompt: str) -> str:
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")

    response = client.interactions.create(
        model=MODEL_NAME,
        input=prompt,
        timeout=60,
    )

    return response.output_text


def generate_research_plan(topic: str) -> ResearchPlan:
    if not topic or not topic.strip():
        raise ValueError("Research topic cannot be empty")

    response = client.interactions.create(
        model=MODEL_NAME,
        input=f"""
        Create a detailed research plan for the following topic:

        {topic}

        Include useful research questions and search queries.
        """,
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": ResearchPlan.model_json_schema(),
        },
        timeout=60,
    )

    return ResearchPlan.model_validate_json(response.output_text)
