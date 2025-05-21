import asyncio
import logging
import os
import json
from datetime import datetime
from dotenv import load_dotenv
import re
from openai import AsyncOpenAI
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import Usage, UsageLimits

from agents import (
    create_answer_generator,
    create_scenario_planning_agent,
    create_question_generator,

    AnswerGeneratorDeps,
    ScenarioPlanningDeps,
    QuestionGeneratorDeps,

    convert_to_json_serializable
)

# Load environment variables
load_dotenv()

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"followup_test_{timestamp}.log")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize DeepInfra-backed OpenAI client and provider
custom_openai_client = AsyncOpenAI(
    api_key=os.getenv("DEEPINFRA_API_KEY"),
    base_url="https://api.deepinfra.com/v1/openai"
)
openai_provider = OpenAIProvider(openai_client=custom_openai_client)

# Sample document chunks for testing
sample_chunks = [
    {
        "chunk_id": "chunk1",
        "content": "Artificial Intelligence (AI) is revolutionizing healthcare delivery across the globe. From diagnostic assistance to personalized treatment plans, AI technologies are enhancing clinical decision-making and improving patient outcomes.",
        "document_title": "sample_document.txt"
    },
    {
        "chunk_id": "chunk2",
        "content": "AI algorithms have demonstrated remarkable accuracy in analyzing medical images. Deep learning models can detect abnormalities in X-rays, MRIs, and CT scans, often matching or exceeding the performance of experienced radiologists.",
        "document_title": "sample_document.txt"
    },
    {
        "chunk_id": "chunk3",
        "content": "The FDA has approved several AI-based diagnostic tools since 2018, including IDx-DR for diabetic retinopathy detection and Viz.AI for stroke detection. These tools are now being integrated into clinical workflows across major healthcare institutions.",
        "document_title": "sample_document.txt"
    }
]

async def test_followup_question_flow(document_chunks=None):
    """
    Test the conversation flow up to the follow-up question stage:
    1. Generate a scenario
    2. Generate the initial question
    3. Generate the initial answer
    4. Generate the follow-up question
    """
    if document_chunks is None:
        document_chunks = sample_chunks
    
    usage = Usage()
    usage_limits = UsageLimits(request_limit=50)
    results = {}

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"followup_test_results_{timestamp}.json")

    # Instantiate agents

    answer_generator = create_answer_generator(openai_provider)
    scenario_planning_agent = create_scenario_planning_agent(provider=openai_provider)
    question_generator = create_question_generator(openai_provider)
    
    try:
        # Step 1: Plan a scenario
        logger.info("Planning conversation scenario")
        scenario_run = await scenario_planning_agent.run(
            "Analyze documents and identify a potential conversation scenario",
            deps=ScenarioPlanningDeps(
                document_chunks=document_chunks,
                num_scenarios=1
            ),
            usage=usage,
            usage_limits=usage_limits
        )
        scenario_result = scenario_run.output
        scenario = scenario_result.scenarios[0]
        
        logger.info(f"Generated scenario: {scenario.title}")
        print(f"\nScenario: {scenario.title}")
        print(f"Persona: {scenario.persona.name} - {scenario.persona.goals}")
        results["scenario"] = convert_to_json_serializable(scenario_result)
        
        # Step 2: Get the initial question from the scenario
        initial_question = scenario.initial_question
        print(f"\nInitial Question: {initial_question}")
        
        # Step 3: Generate answer to the initial question
        logger.info(f"Generating answer for: {initial_question}")
        answer_run = await answer_generator.run(
            initial_question,
            deps=AnswerGeneratorDeps(
                question=initial_question,
                document_chunks=document_chunks,
                max_chunks_to_use=3
            ),
            usage=usage,
            usage_limits=usage_limits
        )
        answer = answer_run.output.answer
        source_ids = answer_run.output.source_chunk_ids
        
        print(f"\nAnswer 0: {answer}")
        print(f"Source chunks: {source_ids}")

        conversation_history = [
                {
                    "role": "user",
                    "content": initial_question
                },
                {
                    "role": "assistant",
                    "content": answer
                }
            ]

        for i in range(3):
        
            # Step 4: Generate follow-up question
            logger.info(f"Generating follow-up question {i+1}")
            
            
            # Create and run the question generator
            question_run = await question_generator.run(
                "Generate a follow-up question",
                deps=QuestionGeneratorDeps(
                    conversation_history=conversation_history,
                    document_chunks=document_chunks,
                    max_chunks_to_use=3
                ),
                usage=usage,
                usage_limits=usage_limits
            )
            
            follow_up_question = question_run.output.question
            related_chunks = question_run.output.related_chunk_ids
            
            print(f"\nFollow-up Question {i+1}: {follow_up_question}")
            print(f"Related chunks: {related_chunks}")

            logger.info(f"Generating answer for: {follow_up_question}")
            answer_run = await answer_generator.run(
                follow_up_question,
                deps=AnswerGeneratorDeps(
                    question=follow_up_question,
                    document_chunks=document_chunks,
                    max_chunks_to_use=5
                ),
                usage=usage,
                usage_limits=usage_limits
            )
            answer = answer_run.output.answer
            source_ids = answer_run.output.source_chunk_ids

            print(f"\nAnswer {i+1}: {answer}")
            print(f"Source chunks: {source_ids}")

            conversation_history.append(
                {
                    "role": "user",
                    "content": follow_up_question
                }
            )
            conversation_history.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )
            results["conversation_history"] = conversation_history
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f)
           
       
        
        # Save results to a file
        
        
        
        
        
        
        logger.info(f"Results saved to: {out_path}")
        logger.info(f"Total tokens used: {usage.total_tokens}")
        
        return conversation_history
        
    except Exception as e:
        logger.error(f"Test error: {e}", exc_info=True)
        return {"error": str(e)}
def process_text_file(
    file_path: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> list[dict[str, any]]:
    """
    Splits a text file into overlapping chunks with metadata.
    """
    logger.info(f"Processing file: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        title = os.path.basename(file_path)
        paragraphs = re.split(r"\n\s*\n", text)

        chunks: list[dict[str, any]] = []
        current, size, idx = "", 0, 1
        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            if size + len(p) > chunk_size and size > 0:
                chunk_id = f"chunk{idx}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "content": current.strip(),
                    "document_title": title
                })
                idx += 1
                if overlap > 0 and len(current) > overlap:
                    current = current[-overlap:] + "\n" + p
                else:
                    current = p
                size = len(current)
            else:
                current = f"{current}\n\n{p}" if current else p
                size = len(current)

        if current:
            chunk_id = f"chunk{idx}"
            chunks.append({
                "chunk_id": chunk_id,
                "content": current.strip(),
                "document_title": title
            })

        logger.info(f"Created {len(chunks)} chunks")
        return chunks

    except Exception as e:
        logger.error(f"Error processing file: {e}", exc_info=True)
        return []
async def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test conversation flow up to follow-up question"
    )
    parser.add_argument("--file", type=str, help="Input text file path (optional)")
    args = parser.parse_args()
    
    document_chunks = None
    if args.file:
        document_chunks = process_text_file(args.file)
        if not document_chunks:
            logger.error("No content to process. Using sample chunks.")
            document_chunks = sample_chunks
    
    await test_followup_question_flow(document_chunks)

if __name__ == "__main__":
    asyncio.run(main())