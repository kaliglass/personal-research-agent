from unittest.mock import Mock, patch

import pytest

from app.llm.client import MODEL_NAME, generate_response


def test_generate_response_success():
    mock_response = Mock()
    mock_response.output_text = "An AI agent can use tools."

    with patch(
        "app.llm.client.client.interactions.create", return_value=mock_response
    ) as mock_create:
        result = generate_response("Explain AI agents")

    assert result == "An AI agent can use tools."

    mock_create.assert_called_once_with(
        model=MODEL_NAME,
        input="Explain AI agents",
        timeout=60,
    )


@pytest.mark.parametrize("prompt", ["", "   ", "\n"])
def test_empty_prompt_rejected(prompt):
    with patch("app.llm.client.client.interactions.create") as mock_create:
        with pytest.raises(ValueError):
            generate_response(prompt)

        mock_create.assert_not_called()
