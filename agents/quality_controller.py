from typing import List, Dict, Any
from pydantic import BaseModel, Field

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel

# Define model classes for the quality controller agent
class QualityControlDeps(BaseModel):
    """Dependencies for the quality controller agent."""
    question: str = Field(..., description="The question")
    answer: str = Field(..., description="The generated answer")
    source_chunks: List[Dict[str, Any]] = Field(..., description="Source chunks used to generate the answer")

class EvaluationResult(BaseModel):
    """Evaluation result from quality controller."""
    factual_accuracy: float = Field(..., ge=0, le=1, description="Score for factual accuracy")
    factual_accuracy_feedback: str = Field(..., description="Feedback on factual accuracy")
    relevance: float = Field(..., ge=0, le=1, description="Score for relevance")
    relevance_feedback: str = Field(..., description="Feedback on relevance")
    naturalness: float = Field(..., ge=0, le=1, description="Score for naturalness")
    naturalness_feedback: str = Field(..., description="Feedback on naturalness")
    overall_score: float = Field(..., ge=0, le=1, description="Overall quality score")
    overall_feedback: str = Field(..., description="Overall feedback")
    passed: bool = Field(..., description="Whether the answer passes quality control")

def create_quality_controller(provider):
    # Initialize the quality controller agent
    quality_controller = Agent(
        OpenAIModel("meta-llama/Llama-3.3-70B-Instruct-Turbo", provider=provider),
        deps_type=QualityControlDeps,
        output_type=EvaluationResult,
        max_result_retries=1,  # Minimize retries to reduce API calls
        system_prompt="""
        You are a quality control expert evaluating answers to questions.
        Assess the conversation for:
        
        1. Factual accuracy: Does the answer contain information that is consistent with the provided document chunks?
        2. Relevance: Is the answer directly addressing the question asked?
        3. Natural conversational flow: Does the conversation sound natural and human-like?
        
        Provide detailed feedback and scores for each criterion, as well as an overall assessment.
        
        IMPORTANT: To minimize API calls, DO NOT use the 'verify_factual_statement' tool.
        Instead, directly evaluate the answer based on the provided source chunks.
        
        ALWAYS use the 'final_result' tool to provide your evaluation results with the following parameters:
          * factual_accuracy: Score from 0 to 1
          * factual_accuracy_feedback: Brief feedback on factual accuracy
          * relevance: Score from 0 to 1
          * relevance_feedback: Brief feedback on relevance
          * naturalness: Score from 0 to 1
          * naturalness_feedback: Brief feedback on naturalness
          * overall_score: Overall quality score from 0 to 1
          * overall_feedback: Brief overall assessment
          * passed: Boolean indicating whether the answer passes quality control
        
        DO NOT provide plain text responses. ALWAYS use the final_result tool for your evaluation.
        """
    )

    @quality_controller.tool
    async def verify_factual_statement(ctx: RunContext[QualityControlDeps], statement: str) -> Dict[str, Any]:
        """
        Verify if a statement from the answer is supported by the source chunks.
        
        Args:
            ctx: The run context containing source chunks
            statement: The statement to verify
            
        Returns:
            Verification result with score and explanation
        """

        # Here we'll use a simple heuristic based on word matching
        statement_words = set(statement.lower().split())
        found_evidence = False
        supporting_text = ""
        
        for chunk in ctx.deps.source_chunks:
            chunk_content = chunk.get("content", "").lower()
            chunk_words = set(chunk_content.split())
            
            # If enough words from the statement appear in the chunk, consider it evidence
            overlap = len(statement_words.intersection(chunk_words)) / len(statement_words) if statement_words else 0
            if overlap > 0.5:
                found_evidence = True
                supporting_text = chunk.get("content", "")
                break
        
        return {
            "statement": statement,
            "verified": found_evidence,
            "score": 1.0 if found_evidence else 0.5,
            "supporting_text": supporting_text
        }
    
    return quality_controller