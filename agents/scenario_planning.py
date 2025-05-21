import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.messages import UserPromptPart

# 1) Dependency model passed into the agent at run time
class ScenarioPlanningDeps(BaseModel):
    document_chunks: List[Dict[str, Any]] = Field(
        ..., description="Chunks of the document to analyze"
    )
    num_scenarios: int = Field(
        4, description="How many scenarios to generate"
    )

# 2) Models for building up the final result
class UserPersona(BaseModel):
    name: Optional[str] = Field(None, description="Persona name")
    type: str = Field(..., description="Persona role/type")
    background: str = Field(..., description="Persona background")
    goals: str = Field(..., description="Persona goals")

class Scenario(BaseModel):
    scenario_id: int = Field(..., description="Unique ID")
    title: str = Field(..., description="Scenario title")
    persona: UserPersona = Field(..., description="User persona")
    context: str = Field(..., description="Situation context")
    initial_question: str = Field(..., description="First user question")
    information_needs: List[str] = Field(..., description="Info needs list")

class ScenarioResult(BaseModel):
    domain: str = Field(..., description="Primary domain")
    topics: List[str] = Field(..., description="Key topics")
    scenarios: List[Scenario] = Field(..., description="Generated scenarios")

# 3) Intermediate models for parsing raw JSON from the LLM
class LLMDomainTopicsResponse(BaseModel):
    domain: str
    topics: List[str]

class LLMPersonaDetail(BaseModel):
    type: str
    background: str
    goals: str

class LLMUserPersonasResponse(BaseModel):
    personas: List[LLMPersonaDetail]

class LLMScenarioDetailsResponse(BaseModel):
    title: str
    context: str
    initial_question: str
    information_needs: List[str]

def create_scenario_planning_agent(
    model: Optional[OpenAIModel] = None,
    provider: Optional[Any] = None
) -> Agent[ScenarioPlanningDeps, ScenarioResult]:
    """
    Create a scenario planning agent with the specified model or provider.
    
    Args:
        model: Optional pre-configured OpenAIModel instance
        provider: Optional provider to use if model is not provided
        
    Returns:
        Configured scenario planning Agent
    """
    # If no model was passed in, use Llama-3.3-70B-Instruct-Turbo via your OpenAIProvider
    if model is None:
        model = OpenAIModel(
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            provider=provider
        )

    # Top‐level agent: wires up deps and final output schema
    agent = Agent(
        model,
        deps_type=ScenarioPlanningDeps,
        output_type=ScenarioResult,
        system_prompt="""
You are an expert at analyzing document content to identify potential conversation scenarios.
Your task is to:
1) Identify the primary domain of the documents.
2) Extract key topics covered.
3) Generate user personas.
4) Create realistic scenarios for each persona.
Return a JSON strictly matching the ScenarioResult schema.
        """
    )

    @agent.tool
    async def extract_domain_topics(
        ctx: RunContext[ScenarioPlanningDeps]
    ) -> Dict[str, Any]:
        """
    Extract the primary domain and key topics from document chunks using an LLM.
    
    Args:
        ctx: The run context containing document chunks.
        
    Returns:
        Dictionary with domain and topics information.
    """
        # Combine up to 10 chunks into one prompt
        chunks = ctx.deps.document_chunks[:10]
        text = "\n\n".join(c.get("content", "") for c in chunks).strip()
        if not text:
            return {"domain": "General", "topics": [], "error": "No content"}
        truncated = text[:4000]

        prompt = f"""
Based on the document below, identify:
1) The primary domain.
2) Up to 5 key topics.

Document:
---
{truncated}
---

Respond with ONLY a valid JSON object with no markdown formatting.
The JSON should have a "domain" field with a string value, and a "topics" field with an array of strings.

For example, if the domain is Finance, and the topics are Investment, Banking, etc., your response should look like:
{{
  "domain": "Finance",
  "topics": ["Investment", "Banking", "Insurance", "Retirement Planning", "Tax"]
}}

Your response:
"""
        # Nested Agent for the raw LLM call
        sub_agent = Agent(model)
        sub_result = await sub_agent.run(prompt)
        raw = sub_result.output

        try:
            # Clean up markdown formatting if present
            cleaned_raw = raw
            if raw.strip().startswith("```") and raw.strip().endswith("```"):
                # Extract content between markdown code blocks
                cleaned_raw = "\n".join(raw.strip().split("\n")[1:-1])
                # Remove language specifier if present
                if cleaned_raw.startswith("json"):
                    cleaned_raw = cleaned_raw[4:].strip()
            
            parsed = LLMDomainTopicsResponse.model_validate_json(cleaned_raw)
            return {
                "domain": parsed.domain,
                "topics": parsed.topics,
                "analyzed_chunks": len(chunks),
                "content_length": len(truncated)
            }
        except ValidationError as e:
            return {"domain": "Error", "topics": [], "error": str(e), "raw_response": raw}

    @agent.tool
    async def generate_user_personas(
        ctx: RunContext[ScenarioPlanningDeps],
        domain: str,
        topics: List[str]
    ) -> List[Dict[str, Any]]:
        """
    Generate potential user personas using an LLM based on document content, domain, and topics.
    
    Args:
        ctx: The run context.
        domain: The identified domain.
        topics: List of key topics.
        
    Returns:
        List of user persona dictionaries.
    """
        n = ctx.deps.num_scenarios
        chunks = ctx.deps.document_chunks[:5]
        text = "\n\n".join(c.get("content", "") for c in chunks).strip()
        truncated = text[:2000]

        prompt = f"""
Domain: {domain}
Topics: {', '.join(topics) if topics else 'None'}
Document sample:
---
{truncated}
---

Generate {n} personas as ONLY a valid JSON object with no markdown formatting.
The JSON should have a "personas" field with an array of objects.
Each object should have "type", "background", and "goals" fields.

For example:
{{
  "personas": [
    {{
      "type": "Radiologist",
      "background": "10 years experience in diagnostic imaging",
      "goals": "Improve diagnostic accuracy using AI tools"
    }},
    {{
      "type": "Hospital Administrator",
      "background": "Managing a 500-bed hospital",
      "goals": "Implement cost-effective AI solutions"
    }}
  ]
}}

Your response:
"""
        sub_agent = Agent(model)
        sub_result = await sub_agent.run(prompt)
        raw = sub_result.output

        try:
            # Clean up markdown formatting if present
            cleaned_raw = raw
            if raw.strip().startswith("```") and raw.strip().endswith("```"):
                # Extract content between markdown code blocks
                cleaned_raw = "\n".join(raw.strip().split("\n")[1:-1])
                # Remove language specifier if present
                if cleaned_raw.startswith("json"):
                    cleaned_raw = cleaned_raw[4:].strip()
            
            parsed = LLMUserPersonasResponse.model_validate_json(cleaned_raw)
            return [
                {
                    "name": f"User {i+1}",
                    "type": p.type,
                    "background": p.background,
                    "goals": p.goals
                }
                for i, p in enumerate(parsed.personas[:n])
            ]
        except ValidationError as e:
            return [{"name": f"ErrorUser{i+1}", "type": "Error", "background": "", "goals": str(e)} for i in range(n)]

    @agent.tool
    async def generate_initial_questions(
        ctx: RunContext[ScenarioPlanningDeps],
        domain: str,
        topics: List[str],
        personas: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
    Generate initial questions and scenario details for each persona using an LLM.
    
    Args:
        ctx: The run context.
        domain: The identified domain.
        topics: List of key topics.
        personas: List of user persona dictionaries.
        
    Returns:
        List of scenario dictionaries.
    """
        scenarios: List[Dict[str, Any]] = []
        chunks = ctx.deps.document_chunks[:3]
        text = "\n\n".join(c.get("content", "") for c in chunks).strip()
        truncated = text[:1000]

        for i, persona in enumerate(personas):
            focus = topics[i % len(topics)] if topics else domain
            prompt = f"""
Persona: {json.dumps(persona)}
Domain: {domain}
Topic focus: {focus}
Content preview:
---
{truncated}
---

Produce ONLY a valid JSON object with no markdown formatting.
The JSON should have "title", "context", "initial_question", and "information_needs" fields.
The "information_needs" field should be an array of strings.

For example:
{{
  "title": "AI Diagnostic Tool Implementation",
  "context": "A hospital is considering adopting new AI diagnostic tools",
  "initial_question": "What are the key benefits of AI diagnostic tools?",
  "information_needs": ["Accuracy rates", "Implementation costs", "Training requirements"]
}}

Your response:
"""
            sub_agent = Agent(model)
            sub_result = await sub_agent.run(prompt)
            raw = sub_result.output

            try:
                # Clean up markdown formatting if present
                cleaned_raw = raw
                if raw.strip().startswith("```") and raw.strip().endswith("```"):
                    # Extract content between markdown code blocks
                    cleaned_raw = "\n".join(raw.strip().split("\n")[1:-1])
                    # Remove language specifier if present
                    if cleaned_raw.startswith("json"):
                        cleaned_raw = cleaned_raw[4:].strip()
                
                details = LLMScenarioDetailsResponse.model_validate_json(cleaned_raw)
                scenarios.append({
                    "scenario_id": i + 1,
                    "title": details.title,
                    "persona": persona,
                    "context": details.context,
                    "initial_question": details.initial_question,
                    "information_needs": details.information_needs
                })
            except ValidationError as e:
                scenarios.append({
                    "scenario_id": i + 1,
                    "title": f"Error {i+1}",
                    "persona": persona,
                    "context": str(e),
                    "initial_question": "",
                    "information_needs": []
                })

        return scenarios

    return agent
