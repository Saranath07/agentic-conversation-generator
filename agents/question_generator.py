import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.messages import UserPromptPart

# 1) Dependencies for the question generator agent
class QuestionGeneratorDeps(BaseModel):
    conversation_history: List[Dict[str, str]] = Field(
        ..., description="Previous conversation messages with 'role' and 'content' fields"
    )
    document_chunks: List[Dict[str, Any]] = Field(
        ..., description="Document chunks to use as context for generating questions"
    )
    max_chunks_to_use: int = Field(
        3, description="Maximum number of chunks to use for context"
    )

# 2) Final output schema
class QuestionGeneratorResult(BaseModel):
    question: str = Field(..., description="The generated follow-up question")
    related_chunk_ids: List[str] = Field(..., description="IDs of related document chunks")

# 3) Intermediate schema for parsing LLM JSON
class LLMQuestionResponse(BaseModel):
    question: str
    related_chunk_ids: List[str]
    reasoning: Optional[str] = None

def create_question_generator(provider) -> Agent[QuestionGeneratorDeps, QuestionGeneratorResult]:
    """
    Creates an agent that generates relevant follow-up questions based on 
    conversation history and document chunks.
    
    Args:
        provider: The provider to use for the LLM
        
    Returns:
        Configured question generator Agent
    """
    # Core LLM model and top-level agent
    model = OpenAIModel("meta-llama/Llama-3.3-70B-Instruct-Turbo", provider=provider)
    question_generator = Agent(
        model,
        deps_type=QuestionGeneratorDeps,
        output_type=QuestionGeneratorResult,
        system_prompt="""
You are an expert question generator that creates relevant, insightful follow-up questions based on conversation history and document content.

Your task is to:
1. Analyze the conversation history to understand the context
2. Examine the document chunks to identify unexplored information
3. Generate a natural follow-up question that:
   - Flows naturally from the previous conversation
   - Explores new information available in the document chunks
   - Is specific and focused (not overly broad)
   - Encourages deeper exploration of the topic

IMPORTANT: You must ONLY generate a question. Do not provide answers or additional information.

Available tool:
- generate_question: Use this to generate a follow-up question based on conversation history and document chunks

DO NOT provide plain text responses. ALWAYS use the generate_question tool. STRICTLY follow the instructions.
        """
    )

    @question_generator.tool
    async def generate_question(
        ctx: RunContext[QuestionGeneratorDeps]
    ) -> Dict[str, Any]:
        """
        Generate a follow-up question based on conversation history and document chunks.
        
        Args:
            ctx: The run context containing conversation history and document chunks
            
        Returns:
            Dictionary with the generated question and related chunk IDs
        """
        conversation_history = ctx.deps.conversation_history
        chunks = ctx.deps.document_chunks
        max_chunks = ctx.deps.max_chunks_to_use
        
        if not chunks or not conversation_history:
            return {
                "question": "Could you tell me more about this topic?",
                "related_chunk_ids": []
            }
        
        # Find relevant chunks based on the conversation
        last_user_message = ""
        last_assistant_message = ""
        
        for msg in reversed(conversation_history):
            if msg["role"] == "user" and not last_user_message:
                last_user_message = msg["content"]
            elif msg["role"] == "assistant" and not last_assistant_message:
                last_assistant_message = msg["content"]
            
            if last_user_message and last_assistant_message:
                break
        
        # Prepare context from chunks
        context = ""
        chunk_ids = []
        for i, chunk in enumerate(chunks[:max_chunks]):
            chunk_id = chunk.get("chunk_id", f"unknown-{i}")
            content = chunk.get("content", "")
            
            if content:
                context += f"\nDocument Chunk {chunk_id}:\n{content}\n"
                chunk_ids.append(chunk_id)
        
        # Create conversation summary
        conversation_summary = "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in conversation_history[-4:]  # Use last 4 messages at most
        ])
        
        prompt = f"""
Generate a natural follow-up question based on the conversation history and document chunks below.

Conversation History:
{conversation_summary}

Document Chunks:
{context}

The follow-up question should:
1. Flow naturally from the previous conversation
2. Explore information in the document chunks that hasn't been covered yet
3. Be specific and focused (not overly broad)
4. Encourage deeper exploration of the topic

Return a JSON object with:
- "question": Your generated follow-up question
- "related_chunk_ids": Array of chunk IDs that contain information related to your question
- "reasoning": Brief explanation of why you chose this question (optional)

Example:
{{
  "question": "How do AI diagnostic tools compare to traditional methods in terms of accuracy?",
  "related_chunk_ids": ["chunk2", "chunk3"],
  "reasoning": "The conversation mentioned AI in healthcare, but hasn't explored the accuracy comparison which is covered in chunk2."
}}

Your response:
"""
        # Instead of using UserPromptPart, pass the prompt content directly
        sub_agent = Agent(model)
        sub_result = await sub_agent.run(user_prompt=prompt.strip())
        raw = sub_result.output
        
        # Clean up markdown formatting if present
        cleaned_raw = raw
        if raw.strip().startswith("```") and raw.strip().endswith("```"):
            # Extract content between markdown code blocks
            cleaned_raw = "\n".join(raw.strip().split("\n")[1:-1])
            # Remove language specifier if present
            if cleaned_raw.startswith("json"):
                cleaned_raw = cleaned_raw[4:].strip()
        
        parsed = LLMQuestionResponse.model_validate_json(cleaned_raw)
        
        return {
            "question": parsed.question,
            "related_chunk_ids": parsed.related_chunk_ids
        }
        

    return question_generator