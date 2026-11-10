from client import generate_response

test_prompts = ["", "  ", "Explain AI agents in three sentences."]

for prompt in test_prompts:
    try:
        response = generate_response(prompt)
        print(f"Success: {response}")

    except ValueError as error:
        print(f"Validation error: {error}")

    except Exception as error:
        print(f"API error: {error}")
