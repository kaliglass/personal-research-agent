import pytest
from pydantic import ValidationError

from app.schemas.research import ResearchPlan


@pytest.fixture
def valid_plan_data():
    return {
        "topic": "Agentic AI Frameworks",
        "objective": "Compare popular agent frameworks",
        "research_questions": [
            "How does LangGraph manage state?",
            "How does CrewAI orchestrate agents?",
        ],
        "search_queries": ["LangGraph architecture", "CrewAI architecture"],
        "depth": "deep",
        "max_sources": 10,
    }


def test_valid_research_plan(valid_plan_data):
    plan = ResearchPlan.model_validate(valid_plan_data)

    assert plan.topic == "Agentic AI Frameworks"
    assert plan.max_sources == 10
    assert plan.depth == "deep"


def test_default_values(valid_plan_data):
    valid_plan_data.pop("depth")
    valid_plan_data.pop("max_sources")

    plan = ResearchPlan.model_validate(valid_plan_data)

    assert plan.depth == "standard"
    assert plan.max_sources == 5


def test_invalid_topic():
    with pytest.raises(ValidationError):
        ResearchPlan(
            topic="AI",
            objective="Compare popular agent frameworks",
            research_questions=["How do agents work?"],
            search_queries=["Agent architecture"],
        )


def test_invalid_depth(valid_plan_data):
    valid_plan_data["depth"] = "extreme"

    with pytest.raises(ValidationError):
        ResearchPlan.model_validate(valid_plan_data)


def test_max_sources_out_of_range(valid_plan_data):
    valid_plan_data["max_sources"] = 100

    with pytest.raises(ValidationError):
        ResearchPlan.model_validate(valid_plan_data)


def test_empty_research_questions(valid_plan_data):
    valid_plan_data["research_questions"] = []

    with pytest.raises(ValidationError):
        ResearchPlan.model_validate(valid_plan_data)


def test_extra_fields_rejected(valid_plan_data):
    valid_plan_data["random_field"] = "hello"

    with pytest.raises(ValidationError):
        ResearchPlan.model_validate(valid_plan_data)
