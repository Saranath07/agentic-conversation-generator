import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.messages import UserPromptPart

# 1) Dependencies for the answer generator agent
class AnswerGeneratorDeps(BaseModel):
    question: str = Field(
        ..., description="Question to answer"
    )
    document_chunks: List[Dict[str, Any]] = Field(
        ..., description="Document chunks to use as context for answering"
    )
    max_chunks_to_use: int = Field(
        5, description="Maximum number of chunks to use for context"
    )

# 2) Final output schema
class AnswerGeneratorResult(BaseModel):
    answer: str = Field(..., description="The generated answer")
    source_chunk_ids: List[str] = Field(..., description="IDs of source chunks used")
    confidence_score: float = Field(
        default=0.0, description="Confidence score (0.0-1.0)"
    )

# 3) Intermediate schemas for parsing LLM JSON
class LLMRelevanceResponse(BaseModel):
    relevance_score: float
    reason: str

class LLMAnswerResponse(BaseModel):
    answer: str
    source_chunk_ids: List[str]
    confidence_score: Optional[float] = None

def create_answer_generator(provider) -> Agent[AnswerGeneratorDeps, AnswerGeneratorResult]:
    """
    Builds and returns an Agent that:
      - Finds relevant chunks for a question
      - Generates an answer based on those chunks
      - Returns the answer with source information
    """
    # Core LLM model and top-level agent
    model = OpenAIModel("meta-llama/Llama-3.3-70B-Instruct-Turbo", provider=provider)
    answer_generator = Agent(
        model,
        deps_type=AnswerGeneratorDeps,
        output_type=AnswerGeneratorResult,
        system_prompt="""
You are an expert answer generator that creates accurate, helpful answers based on provided document chunks.

Your answers should:
1. Be directly based on the information in the provided chunks
2. Be comprehensive but concise
3. Cite the source chunks used
4. Maintain a helpful, informative tone

IMPORTANT: You MUST use the provided tools in the following sequence:

1. First, use the 'find_relevant_chunks' tool to identify the most relevant chunks for the question
2. Then, use the 'generate_answer' tool to create an answer based on those chunks
3. Finally, use the 'final_result' tool to provide your final answer with sources

Available tools:
- find_relevant_chunks: Use this to identify chunks relevant to the question
- generate_answer: Use this to generate an answer from relevant chunks
- final_result: ALWAYS use this tool to provide your final answer

DO NOT provide plain text responses. ALWAYS use the appropriate tool for each step.
        """
    )

    @answer_generator.tool
    async def find_relevant_chunks(
        ctx: RunContext[AnswerGeneratorDeps]
    ) -> List[Dict[str, Any]]:
        """
    Find chunks that are most relevant to the question.
    
    Args:
        ctx: The run context containing the question and document chunks
        
    Returns:
        List of the most relevant chunks
    """
        question = ctx.deps.question
        chunks = ctx.deps.document_chunks
        max_chunks = ctx.deps.max_chunks_to_use
        
        if not chunks:
            return []
        
        # Score each chunk for relevance
        scored_chunks = []
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", "unknown")
            content = chunk.get("content", "")
            
            if not content:
                continue
                
            truncated = content[:1000]  # Limit content size
            prompt = f"""
Rate the relevance of this document chunk to the question on a scale of 0.0 to 1.0.

Question: {question}

Document Chunk:
---
{truncated}
---

Return a JSON object with keys "relevance_score" (float between 0.0 and 1.0) and "reason" (brief explanation).
"""
            messages = [UserPromptPart(content=prompt.strip())]
            
            try:
                sub_agent = Agent(model)
                sub_result = await sub_agent.run(message_history=messages)
                raw = sub_result.output
                parsed = LLMRelevanceResponse.model_validate_json(raw)
                
                scored_chunks.append({
                    "chunk": chunk,
                    "score": parsed.relevance_score,
                    "reason": parsed.reason
                })
            except Exception as e:
                # On error, assign a low score but keep the chunk
                scored_chunks.append({
                    "chunk": chunk,
                    "score": 0.1,
                    "reason": f"Error scoring: {str(e)}"
                })
        
        # Sort by relevance score and take top N
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        top_chunks = [item["chunk"] for item in scored_chunks[:max_chunks]]
        
        return top_chunks

    @answer_generator.tool
    async def generate_answer(
        ctx: RunContext[AnswerGeneratorDeps],
        relevant_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
    Generate an answer based on the relevant chunks.
    
    Args:
        ctx: The run context
        relevant_chunks: List of relevant document chunks
        
    Returns:
        Dictionary with the answer and source information
    """
        question = ctx.deps.question
        
        if not relevant_chunks:
            return {
                "answer": "I don't have enough information to answer this question.",
                "source_chunk_ids": [],
                "confidence_score": 0.0
            }
        
        # Prepare context from chunks
        context = ""
        chunk_ids = []
        for i, chunk in enumerate(relevant_chunks):
            chunk_id = chunk.get("chunk_id", f"unknown-{i}")
            content = chunk.get("content", "")
            title = chunk.get("document_title", "Untitled")
            
            if content:
                context += f"\nSource {chunk_id} ({title}):\n{content}\n"
                chunk_ids.append(chunk_id)
        
        prompt = f"""
Answer the following question based ONLY on the provided sources.
If you cannot answer from the sources, say so clearly.

Question: {question}

Sources:
{context}

Return a JSON object with:
- "answer": Your complete answer
- "source_chunk_ids": Array of chunk IDs you used (e.g. ["chunk1", "chunk3"])
- "confidence_score": Your confidence from 0.0 to 1.0

Example:
{{
  "answer": "The detailed answer to the question...",
  "source_chunk_ids": ["chunk1", "chunk3"],
  "confidence_score": 0.85
}}
"""
        messages = [UserPromptPart(content=prompt.strip())]
        
        try:
            sub_agent = Agent(model)
            sub_result = await sub_agent.run(message_history=messages)
            raw = sub_result.output
            parsed = LLMAnswerResponse.model_validate_json(raw)
            
            return {
                "answer": parsed.answer,
                "source_chunk_ids": parsed.source_chunk_ids,
                "confidence_score": parsed.confidence_score or 0.0
            }
        except ValidationError as e:
            # If JSON parsing fails, try to extract just the answer
            return {
                "answer": raw[:1000] if raw else "Error generating answer.",
                "source_chunk_ids": chunk_ids,
                "confidence_score": 0.0,
                "error": str(e)
            }
        except Exception as e:
            return {
                "answer": "An error occurred while generating the answer.",
                "source_chunk_ids": chunk_ids,
                "confidence_score": 0.0,
                "error": str(e)
            }

    return answer_generator