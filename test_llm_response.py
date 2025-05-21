import asyncio
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# 1) Define your dependency model
class ScenarioPlanningDeps(BaseModel):
    document_chunks: list[dict[str, str]]
    num_scenarios: int

async def test_llm_response():
    # 2) Load your DeepInfra API key from .env
    load_dotenv()
    api_key = os.getenv("DEEPINFRA_API_KEY")
    if not api_key:
        raise ValueError("DEEPINFRA_API_KEY not found. Please set it in your .env file.")

    # 3) Initialize a custom AsyncOpenAI client against DeepInfra
    custom_client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.deepinfra.com/v1/openai"
    )
    provider = OpenAIProvider(openai_client=custom_client)

    # 4) Create the PydanticAI model with your custom provider
    model = OpenAIModel(
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",  # replace with your model
        provider=provider
    )

    # 5) Wrap the model in an Agent
    agent = Agent(model)

    # 6) Instantiate your deps
    deps = ScenarioPlanningDeps(
        document_chunks=[{"content": "This is a test document about AI and machine learning."}],
        num_scenarios=1
    )

    # 7) Define your user prompt
    prompt_text = (
        "Summarize the following text in one sentence: "
        "'AI is transforming many industries.'"
    )

    # 8) Call agent.run() with user_prompt and deps
    result = await agent.run(
        user_prompt=prompt_text,
        deps=deps
    )

    # 9) Inspect and print the result
    print("Output summary:", result.output)
    print("Token usage:", result.usage)

if __name__ == "__main__":
    asyncio.run(test_llm_response())
