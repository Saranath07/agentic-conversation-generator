from typing import List, Dict, Any
from pydantic import BaseModel, Field

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel

# Define model classes for the answer generator agent
class AnswerGeneratorDeps(BaseModel):
    """Dependencies for the answer generator agent."""
    question: str = Field(..., description="Question to answer")
    document_chunks: List[Dict[str, Any]] = Field(..., description="Document chunks to use for answering")
    max_chunks_to_use: int = Field(5, description="Maximum number of chunks to use for answering")

class GeneratedAnswer(BaseModel):
    """An answer generated for a question."""
    question: str = Field(..., description="The original question")
    answer: str = Field(..., description="The generated answer")
    source_chunk_ids: List[str] = Field(..., description="IDs of source chunks used")

def create_answer_generator(provider):
    # Initialize the answer generator agent
    answer_generator = Agent(
        OpenAIModel("meta-llama/Llama-3.3-70B-Instruct-Turbo", provider=provider),
        output_type=GeneratedAnswer,
        system_prompt="""
        You are an expert at providing helpful, accurate answers based on document content.
        Your answers should be:
        
        1. Directly based on the information in the provided document chunks
        2. Comprehensive but concise
        3. Written in a natural, conversational tone
        4. Factually accurate and properly cited
        
        Only use information from the provided document chunks. If the answer cannot be found in the chunks,
        clearly state that the information is not available in the provided documents.
        """
    )

    @answer_generator.tool
    async def retrieve_relevant_chunks(ctx: RunContext[AnswerGeneratorDeps], question: str) -> List[Dict[str, Any]]:
        """
        Retrieve chunks that are most relevant to answering the question.
        
        Args:
            ctx: The run context containing all available document chunks
            question: The question to find relevant chunks for
            
        Returns:
            List of relevant document chunks
        """
        # In a real implementation, this would use embeddings and semantic search
        # For this example, we'll just return a subset of the available chunks
        max_chunks = min(ctx.deps.max_chunks_to_use, len(ctx.deps.document_chunks))
        return ctx.deps.document_chunks[:max_chunks]

    @answer_generator.tool
    async def format_answer(ctx: RunContext[AnswerGeneratorDeps], draft_answer: str) -> str:
        """
        Format and improve the draft answer to make it more conversational and readable.
        
        Args:
            ctx: The run context
            draft_answer: The initial draft answer
            
        Returns:
            Formatted and improved answer
        """
        # In a real implementation, this might call a separate model
        # Here we'll just add some simple formatting
        improved_answer = draft_answer.replace("\n\n", "\n")
        
        # Add a conversational opener
        conversational_openers = [
            "Based on the information I have, ",
            "From what I can see, ",
            "According to the documents, ",
        ]
        import random
        opener = random.choice(conversational_openers)
        
        return f"{opener}{improved_answer}"
    
    return answer_generator