from __future__ import annotations
import json
import logging
import asyncio
import os
import re
import uuid
from typing import List, Dict, Any
from datetime import datetime

from openai import AsyncOpenAI
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import Usage, UsageLimits
from dotenv import load_dotenv

from agents import (
    create_question_generator,
    create_answer_generator,
    create_quality_controller,
    create_scenario_planning_agent,
    QuestionGeneratorDeps,
    AnswerGeneratorDeps,
    QualityControlDeps,
    ScenarioPlanningDeps,
    convert_to_json_serializable
)

# Load environment variables
load_dotenv()

# Configure detailed logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"conversation_pipeline_{timestamp}.log")

# Configure logging to file and console with detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure OpenAI library logging
openai_logger = logging.getLogger("openai")
openai_logger.setLevel(logging.DEBUG)

# Initialize OpenAI client with DeepInfra API
custom_openai_client = AsyncOpenAI(
    api_key=os.getenv("DEEPINFRA_API_KEY"),
    base_url="https://api.deepinfra.com/v1/openai"
)

# Create OpenAI provider
openai_provider = OpenAIProvider(openai_client=custom_openai_client)

# Model name
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

async def generate_follow_up_question(
    question_generator,
    conversation_history: List[Dict[str, str]],
    document_chunks: List[Dict[str, Any]],
    usage: Usage,
    usage_limits: UsageLimits
) -> str:
    """
    Generate a follow-up question based on the conversation history.
    
    Args:
        question_generator: The question generator agent
        conversation_history: List of previous questions and answers
        document_chunks: List of document chunks
        usage: Usage tracking object
        usage_limits: Usage limits
        
    Returns:
        A follow-up question
    """
    # Get the most recent question and answer
    previous_question = conversation_history[-1]["question"]
    previous_answer = conversation_history[-1]["answer"]
    
    logger.info(f"Generating follow-up question based on conversation history with {len(conversation_history)} exchanges")
    
    # Create a context that includes the full conversation history
    context = "Conversation history:\n"
    for i, exchange in enumerate(conversation_history):
        context += f"Q{i+1}: {exchange['question']}\n"
        context += f"A{i+1}: {exchange['answer']}\n\n"
    
    # Enhanced prompt to ensure follow-up questions are natural continuations
    # and not tied to any specific scenario
    context += """
Based on this conversation history, generate a natural follow-up question that:
1. Continues the conversation flow naturally
2. Explores a logical next topic based on the previous answer
3. Maintains a conversational tone
4. Is not tied to any specific scenario beyond what's already discussed
5. Encourages further exploration of the topic

The follow-up question should feel like it comes from the same person who asked the previous questions.
"""
    
    try:
        # Use the question generator to create a follow-up question
        question_result = await question_generator.run(
            context,  # Provide context through the prompt
            deps=QuestionGeneratorDeps(
                document_chunks=document_chunks,
                questions_per_chunk=1  # We only need one follow-up question
            ),
            usage=usage,
            usage_limits=usage_limits,
            debug=True
        )
        
        if question_result.output.questions:
            follow_up_question = question_result.output.questions[0].question
            logger.info(f"Generated follow-up question: {follow_up_question}")
            return follow_up_question
        else:
            logger.warning("No follow-up question was generated")
            return "Can you tell me more about that?"
            
    except Exception as e:
        logger.error(f"Error generating follow-up question: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return "Can you elaborate on that further?"

async def run_conversation_pipeline(
    document_chunks: List[Dict[str, Any]],
    usage: Usage = None,
    conversation_rounds: int = 3  # Default to 3 rounds of conversation
) -> Dict[str, Any]:
    """
    Run the complete conversation pipeline with multi-round conversations.
    
    The pipeline works as follows:
    1. Generate 3-4 scenarios based on document content
    2. For each scenario, use its initial question to start a conversation
    3. Generate answers for each question
    4. For follow-up questions, generate questions based solely on conversation flow
       (not tied to specific scenarios)
    5. Evaluate the quality of each answer
    6. Return the complete conversation results
    
    Args:
        document_chunks: List of document chunks to process
        usage: Optional usage tracking object
        conversation_rounds: Number of rounds in each conversation (question-answer pairs)
        
    Returns:
        Dictionary with results from all stages of the pipeline
    """
    usage = usage or Usage()
    usage_limits = UsageLimits(request_limit=50)
    
    # Log the document chunks for debugging
    logger.debug(f"Document chunks: {json.dumps([{k: v for k, v in chunk.items() if k != 'content'} for chunk in document_chunks], indent=2)}")
    for i, chunk in enumerate(document_chunks):
        logger.debug(f"Chunk {i} ID: {chunk.get('chunk_id')}, Title: {chunk.get('document_title')}")
        logger.debug(f"Chunk {i} content preview: {chunk.get('content', '')[:100]}...")
    
    logger.info(f"Starting conversation pipeline with {conversation_rounds} rounds per conversation")
    results = {}
    
    try:
        # Initialize agents
        question_generator = create_question_generator(openai_provider)
        answer_generator = create_answer_generator(openai_provider)
        quality_controller = create_quality_controller(openai_provider)
        scenario_planning_agent = create_scenario_planning_agent(openai_provider)
        
        # Step 1: Generate scenarios for conversations (only for initial questions)
        logger.info("Planning conversation scenarios")
        try:
            # Use the scenario planning agent to generate scenarios
            scenario_result = await scenario_planning_agent.run(
                "Analyze documents and identify potential conversation scenarios",
                deps=ScenarioPlanningDeps(
                    document_chunks=document_chunks,
                    num_scenarios=4  # Generate 3-4 scenarios as requested
                ),
                usage=usage,
                usage_limits=usage_limits,
                debug=True  # Enable debug mode
            )
            
            
            results["scenarios"] = scenario_result.output.model_dump()
            logger.info(f"Generated {len(scenario_result.output.scenarios)} conversation scenarios")
        except Exception as e:
            logger.error(f"Error in scenario planning: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        
        # Step 2: For each scenario, generate multi-round conversations
        logger.info("Generating multi-round conversations for each scenario")
        conversations = []
        simplified_conversations = []
        
        for scenario in scenario_result.output.scenarios:
            logger.info(f"Processing scenario: {scenario.title}")
            
            # Start a new conversation for this scenario
            conversation = {
                "scenario_id": scenario.scenario_id,
                "scenario_title": scenario.title,
                "persona": scenario.persona.model_dump(),
                "rounds": []
            }
            
            # Also create a simplified conversation structure
            simplified_conversation = {
                "scenario": scenario.title,
                "exchanges": []
            }
            
            # Use the initial question from the scenario to start the conversation
            current_question = scenario.initial_question
            
            # Keep track of conversation history for context
            conversation_history = []
            
            # Generate multiple rounds of conversation
            for round_num in range(conversation_rounds):
                logger.info(f"Conversation round {round_num + 1} for scenario {scenario.scenario_id}")
                
                # Generate answer for the current question
                try:
                    answer_result = await answer_generator.run(
                        current_question,
                        deps=AnswerGeneratorDeps(
                            question=current_question,
                            document_chunks=document_chunks,
                            max_chunks_to_use=5
                        ),
                        usage=usage,
                        usage_limits=usage_limits,
                        debug=True
                    )
                    
                    current_answer = answer_result.output.answer
                    source_chunk_ids = answer_result.output.source_chunk_ids
                    
                    # Evaluate answer quality
                    source_chunks = [chunk for chunk in document_chunks if chunk.get("chunk_id") in source_chunk_ids]
                    evaluation_result = await quality_controller.run(
                        f"Evaluate this Q&A pair: Question: {current_question} Answer: {current_answer}",
                        deps=QualityControlDeps(
                            question=current_question,
                            answer=current_answer,
                            source_chunks=source_chunks
                        ),
                        usage=usage,
                        usage_limits=usage_limits,
                        debug=True
                    )
                    
                    # Add this Q&A pair to the conversation
                    conversation["rounds"].append({
                        "round": round_num + 1,
                        "question": current_question,
                        "answer": current_answer,
                        "source_chunk_ids": source_chunk_ids,
                        "quality_evaluation": evaluation_result.output.model_dump()
                    })
                    
                    # Add to conversation history for context
                    conversation_history.append({
                        "question": current_question,
                        "answer": current_answer
                    })
                    
                    # Add to simplified conversation
                    simplified_conversation["exchanges"].append({
                        "question": current_question,
                        "answer": current_answer
                    })
                    
                    # If this isn't the last round, generate a follow-up question
                    if round_num < conversation_rounds - 1:
                        # Generate follow-up question based only on conversation history
                        # This ensures follow-up questions are not tied to specific scenarios
                        # but are natural continuations of the conversation
                        current_question = await generate_follow_up_question(
                            question_generator,
                            conversation_history,
                            document_chunks,
                            usage,
                            usage_limits
                        )
                    
                except Exception as e:
                    logger.error(f"Error in conversation round {round_num + 1}: {str(e)}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    # Add partial round information if available
                    if 'current_question' in locals():
                        conversation["rounds"].append({
                            "round": round_num + 1,
                            "question": current_question,
                            "error": str(e)
                        })
                    break
            
            conversations.append(conversation)
            simplified_conversations.append(simplified_conversation)
        
        results["conversations"] = conversations
        results["simplified_conversations"] = simplified_conversations
        logger.info(f"Generated {len(conversations)} multi-round conversations")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in conversation pipeline: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"error": str(e), "partial_results": results}

def process_text_file(file_path: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Process a text file into document chunks suitable for the conversation pipeline.
    
    Args:
        file_path: Path to the text file
        chunk_size: Maximum number of characters per chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of document chunks with chunk_id, content, and document_title
    """
    logger.info(f"Processing text file: {file_path}")
    
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Extract document title from filename
        document_title = os.path.basename(file_path)
        
        # Split text into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = ""
        current_size = 0
        chunk_index = 1
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # If adding this paragraph would exceed chunk size, save current chunk and start a new one
            if current_size + len(paragraph) > chunk_size and current_size > 0:
                # Create a chunk with a predictable ID
                chunk_id = f"chunk{chunk_index}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "content": current_chunk.strip(),
                    "document_title": document_title
                })
                chunk_index += 1
                
                # Start new chunk with overlap from previous chunk
                if overlap > 0 and len(current_chunk) > overlap:
                    current_chunk = current_chunk[-overlap:] + "\n" + paragraph
                else:
                    current_chunk = paragraph
                current_size = len(current_chunk)
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                current_size = len(current_chunk)
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunk_id = f"chunk{chunk_index}"
            chunks.append({
                "chunk_id": chunk_id,
                "content": current_chunk.strip(),
                "document_title": document_title
            })
        
        logger.info(f"Created {len(chunks)} chunks from {file_path}")
        
        # Log the first few chunks for debugging
        for i, chunk in enumerate(chunks[:3]):
            logger.debug(f"Sample chunk {i+1}:")
            logger.debug(f"  ID: {chunk['chunk_id']}")
            logger.debug(f"  Title: {chunk['document_title']}")
            logger.debug(f"  Content preview: {chunk['content'][:100]}...")
            
        return chunks
        
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return []

async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run conversation pipeline on a text file')
    parser.add_argument('--file', type=str, help='Path to the text file to process')
    parser.add_argument('--chunk-size', type=int, default=500, help='Maximum characters per chunk')
    parser.add_argument('--overlap', type=int, default=50, help='Character overlap between chunks')
    parser.add_argument('--demo', action='store_true', help='Run with demo data instead of file')
    parser.add_argument('--rounds', type=int, default=3, help='Number of conversation rounds per scenario')
    
    args = parser.parse_args()
    
    # Track API usage
    usage = Usage()
    
    if args.demo:
        logger.info("Running with demo document chunks")
        # Example document chunks for demo
        document_chunks = [
            {
                "chunk_id": "chunk1",
                "content": "Environmental regulations require companies to report emissions quarterly.",
                "document_title": "Emission Regulations Guide"
            },
            {
                "chunk_id": "chunk2",
                "content": "Penalties for non-compliance can range from $1,000 to $50,000 per day.",
                "document_title": "Emission Regulations Guide"
            },
            {
                "chunk_id": "chunk3",
                "content": "Companies must install monitoring equipment that meets the EPA standards.",
                "document_title": "Implementation Guidelines"
            }
        ]
    elif args.file:
        logger.info(f"Processing file: {args.file}")
        document_chunks = process_text_file(args.file, args.chunk_size, args.overlap)
        if not document_chunks:
            logger.error("Failed to process file or no content found")
            return
    else:
        logger.error("No input provided. Use --file to specify a text file or --demo to use demo data")
        return
    
    logger.info(f"Running conversation pipeline with {len(document_chunks)} document chunks and {args.rounds} conversation rounds")
    results = await run_conversation_pipeline(document_chunks, usage, args.rounds)
    
    # Save results to file
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full results
    output_file = os.path.join(output_dir, f"conversation_results_{timestamp}.json")
    json_results = convert_to_json_serializable(results)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(obj=json_results, fp=f, indent=2)
    
    # Save simplified conversations (just question-answer pairs)
    simplified_output_file = os.path.join(output_dir, f"simplified_conversations_{timestamp}.json")
    simplified_json_results = convert_to_json_serializable(results["simplified_conversations"])
    with open(simplified_output_file, 'w', encoding='utf-8') as f:
        json.dump(obj=simplified_json_results, fp=f, indent=2)
    
    logger.info(f"Full results saved to {output_file}")
    logger.info(f"Simplified conversations saved to {simplified_output_file}")
    
    # Log API usage statistics
    logger.info(f"Total tokens: {usage.total_tokens}")
    # Log other available usage metrics
    for attr in dir(usage):
        if not attr.startswith('_') and attr != 'total_tokens':
            try:
                value = getattr(usage, attr)
                if isinstance(value, (int, float)):
                    logger.info(f"{attr}: {value}")
            except:
                pass
    
    print(json.dumps(json_results, indent=2))

if __name__ == "__main__":
    asyncio.run(main())