from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ResearchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    topic: str = Field(
        min_length=3, max_length=200, description="The topic being researched"
    )

    objective: str = Field(
        min_length=10, description="The main objective of the research"
    )

    research_questions: list[str] = Field(min_length=1, max_length=8)

    search_queries: list[str] = Field(min_length=1, max_length=10)

    depth: Literal["quick", "standard", "deep"] = "standard"

    max_sources: int = Field(default=5, ge=1, le=20)
