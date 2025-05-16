from typing import List, Dict, Any
from pydantic import BaseModel, Field

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel

# Define model classes for the question generator agent
class QuestionGeneratorDeps(BaseModel):
    """Dependencies for the question generator agent."""
    document_chunks: List[Dict[str, Any]] = Field(..., description="Document chunks to generate questions from")
    questions_per_chunk: int = Field(3, description="Number of questions to generate per chunk")

class GeneratedQuestion(BaseModel):
    """A question generated from document content."""
    question: str = Field(..., description="The generated question")
    source_chunk_id: str = Field(..., description="ID of the source chunk")
    document_title: str = Field(..., description="Title of the source document")

class QuestionGeneratorResult(BaseModel):
    """Result from question generator agent."""
    questions: List[GeneratedQuestion] = Field(..., description="List of generated questions")

def create_question_generator(provider):
    # Initialize the question generator agent
    question_generator = Agent(
        OpenAIModel("meta-llama/Llama-3.3-70B-Instruct-Turbo", provider=provider),
        deps_type=QuestionGeneratorDeps,
        output_type=QuestionGeneratorResult,
        system_prompt="""
        You are an expert question generator that creates natural, conversational questions based on document chunks.
        Your questions should:
        
        1. Be directly answerable from the provided content
        2. Cover different aspects of the document
        3. Sound natural and conversational, not academic or formal
        4. Be specific enough to be answered with the information provided
        
        Generate diverse questions that would help users understand the key points in the documents.
        """
    )

    @question_generator.tool
    async def analyze_document_chunk(ctx: RunContext[QuestionGeneratorDeps], chunk_id: str) -> Dict[str, Any]:
        """
        Analyze a specific document chunk to extract key topics and information.
        
        Args:
            ctx: The run context containing document chunks
            chunk_id: ID of the chunk to analyze
            
        Returns:
            Dictionary with analyzed information about the chunk
        """
        for chunk in ctx.deps.document_chunks:
            if chunk.get("chunk_id") == chunk_id:
                # In a real implementation, you might perform more sophisticated analysis
                topics = ["Topic extracted from document"]  # Placeholder
                return {
                    "chunk_id": chunk_id,
                    "content": chunk.get("content", ""),
                    "document_title": chunk.get("document_title", "Untitled"),
                    "key_topics": topics
                }
        
        return {
            "chunk_id": chunk_id,
            "error": "Chunk not found",
            "content": "",
            "document_title": "Unknown",
            "key_topics": []
        }

    @question_generator.tool
    async def humanize_question(ctx: RunContext[QuestionGeneratorDeps], question: str) -> str:
        """
        Make a question sound more natural and conversational.
        
        Args:
            ctx: The run context
            question: Question to humanize
            
        Returns:
            Humanized version of the question
        """
        # In a real implementation, this might call a separate model or API
        # Here we'll just add a simple modification to the question
        conversational_starters = [
            "I was wondering, ",
            "Could you tell me ",
            "I'm curious about ",
            "I'd like to know ",
        ]
        import random
        starter = random.choice(conversational_starters)
        
        # Make sure the question ends with a question mark
        if not question.endswith('?'):
            question += '?'
        
        # Remove any existing question marks before adding the starter
        question = question.rstrip('?')
        
        return f"{starter}{question.lower()}?"
    
    return question_generator