from app.llm.client import generate_research_plan


def test_generate_research_plan():
    plan = generate_research_plan("AI Agent Frameworks")

    assert plan.topic
    assert len(plan.research_questions) >= 1
    assert len(plan.search_queries) >= 1
    assert plan.depth in ["quick", "standard", "deep"]

    print(plan.model_dump_json(indent=2))
