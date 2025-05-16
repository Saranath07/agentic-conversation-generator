from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel

# Define model classes for the scenario planning agent
class ScenarioPlanningDeps(BaseModel):
    """Dependencies for the scenario planning agent."""
    document_chunks: List[Dict[str, Any]] = Field(..., description="Document chunks to analyze")
    num_scenarios: int = Field(4, description="Number of scenarios to generate")

class UserPersona(BaseModel):
    """A user persona for a scenario."""
    name: Optional[str] = Field(None, description="Name of the persona")
    type: str = Field(..., description="Type of user")
    background: str = Field(..., description="Background of the user")
    goals: str = Field(..., description="User's goals")

class Scenario(BaseModel):
    """A conversation scenario."""
    scenario_id: int = Field(..., description="Unique identifier for the scenario")
    title: str = Field(..., description="Brief descriptive title for the scenario")
    persona: UserPersona = Field(..., description="User persona for the scenario")
    context: str = Field(..., description="The specific context or situation")
    initial_question: str = Field(..., description="The first question the user would ask")
    information_needs: List[str] = Field(..., description="Specific information needs")

class ScenarioResult(BaseModel):
    """Result from scenario planning agent."""
    domain: str = Field(..., description="The primary domain identified")
    topics: List[str] = Field(..., description="Key topics identified")
    scenarios: List[Scenario] = Field(..., description="Generated scenarios")

def create_scenario_planning_agent(provider):
    # Initialize the scenario planning agent
    scenario_planning_agent = Agent(
        OpenAIModel("meta-llama/Llama-3.3-70B-Instruct-Turbo", provider=provider),
        deps_type=ScenarioPlanningDeps,
        output_type=ScenarioResult,
        system_prompt="""
        You are an expert at analyzing document content to identify potential conversation scenarios.
        Your task is to:
        
        1. Identify the primary domain or field of the documents
        2. Extract key topics or subjects covered in the content
        3. Identify potential types of users who might be interested in this content
        4. Create realistic conversation scenarios for different user personas
        
        Each scenario should include a specific user persona, the context of their inquiry,
        their initial question, and their information needs.
        """
    )

    @scenario_planning_agent.tool
    async def extract_domain_topics(ctx: RunContext[ScenarioPlanningDeps]) -> Dict[str, Any]:
        """
        Extract the primary domain and key topics from document chunks.
        
        Args:
            ctx: The run context containing document chunks
            
        Returns:
            Dictionary with domain and topics information
        """
        # Get a representative sample of chunks for analysis
        sample_size = min(10, len(ctx.deps.document_chunks))
        sample_chunks = ctx.deps.document_chunks[:sample_size]
        
        # Combine content from all sample chunks for analysis
        combined_content = ""
        for chunk in sample_chunks:
            content = chunk.get("content", "")
            if content:
                combined_content += content + "\n\n"
        
        # Extract titles for additional context
        titles = [chunk.get("document_title", "") for chunk in sample_chunks if "document_title" in chunk]
        unique_titles = list(set(titles))
        
        # Extract potential domain from titles and headings
        domain = "General"
        if combined_content:
            # Look for headings that might indicate domain
            import re
            headings = re.findall(r'#+ (.+)', combined_content)
            if headings:
                # Use the first main heading as a potential domain indicator
                domain = headings[0].strip()
            elif unique_titles:
                # Use document title if no headings found
                domain = unique_titles[0]
        
        # Extract potential topics from content
        topics = []
        if combined_content:
            # Look for subheadings as potential topics
            subheadings = re.findall(r'#{2,} (.+)', combined_content)
            if subheadings:
                topics = [heading.strip() for heading in subheadings[:5]]  # Limit to top 5 topics
            
            # If no subheadings, try to extract key phrases
            if not topics:
                # Simple keyword extraction based on frequency
                words = re.findall(r'\b[A-Za-z][A-Za-z-]+\b', combined_content.lower())
                # Filter out common words
                common_words = {'the', 'and', 'for', 'with', 'that', 'this', 'are', 'from', 'have', 'has', 'not'}
                filtered_words = [word for word in words if word not in common_words and len(word) > 3]
                
                # Count word frequency
                from collections import Counter
                word_counts = Counter(filtered_words)
                
                # Get top 5 most common words as topics
                topics = [word.capitalize() for word, _ in word_counts.most_common(5)]
        
        # Ensure we have at least some topics
        if not topics:
            # Extract nouns from the first paragraph as fallback
            first_para = combined_content.split('\n\n')[0] if combined_content else ""
            words = re.findall(r'\b[A-Za-z][A-Za-z-]+\b', first_para)
            topics = [word.capitalize() for word in words[:5] if len(word) > 3]
        
        return {
            "domain": domain,
            "topics": topics,
            "analyzed_chunks": sample_size,
            "content_sample": combined_content[:200] + "..." if len(combined_content) > 200 else combined_content
        }

    @scenario_planning_agent.tool
    async def generate_user_personas(ctx: RunContext[ScenarioPlanningDeps], domain: str, topics: List[str]) -> List[Dict[str, Any]]:
        """
        Generate potential user personas based on document content, domain and topics.
        
        Args:
            ctx: The run context containing document chunks
            domain: The identified domain
            topics: List of key topics
            
        Returns:
            List of user persona dictionaries
        """
        # Get a representative sample of document chunks for analysis
        sample_size = min(10, len(ctx.deps.document_chunks))
        sample_chunks = ctx.deps.document_chunks[:sample_size]
        
        # Combine content from all sample chunks for analysis
        combined_content = ""
        for chunk in sample_chunks:
            content = chunk.get("content", "")
            if content:
                combined_content += content + "\n\n"
        
        # Extract potential stakeholders and user types from the content
        import re
        
        # Look for mentions of specific roles, professions, or stakeholders
        potential_stakeholders = []
        
        # Common role patterns
        role_patterns = [
            r'(?:for|by|to) (\w+(?:\s\w+){0,2}) (?:professionals|specialists|experts)',
            r'(?:for|by|to) (\w+(?:\s\w+){0,2}) (?:in|of|at)',
            r'(\w+(?:\s\w+){0,2}) (?:can|could|may|might|should|would) (?:use|benefit|consider)',
            r'(\w+(?:\s\w+){0,2}) (?:needs|requires|demands)',
            r'(?:help|assist|enable) (\w+(?:\s\w+){0,2}) (?:to|in|with)'
        ]
        
        for pattern in role_patterns:
            matches = re.findall(pattern, combined_content)
            potential_stakeholders.extend(matches)
        
        # If we couldn't extract specific stakeholders, infer them from the domain and topics
        if not potential_stakeholders:
            # Analyze the domain to determine potential stakeholders
            if "healthcare" in domain.lower() or any("health" in topic.lower() for topic in topics):
                potential_stakeholders = ["Healthcare Provider", "Patient", "Medical Researcher", "Hospital Administrator"]
            elif "education" in domain.lower() or any("education" in topic.lower() for topic in topics):
                potential_stakeholders = ["Teacher", "Student", "School Administrator", "Education Researcher"]
            elif "technology" in domain.lower() or any(tech_term in " ".join(topics).lower() for tech_term in ["ai", "software", "data", "computer", "technology"]):
                potential_stakeholders = ["Software Developer", "Data Scientist", "IT Manager", "Technology Consultant"]
            elif "business" in domain.lower() or any(biz_term in " ".join(topics).lower() for biz_term in ["business", "market", "finance", "economy"]):
                potential_stakeholders = ["Business Owner", "Manager", "Consultant", "Investor"]
            elif "environment" in domain.lower() or any(env_term in " ".join(topics).lower() for env_term in ["environment", "climate", "sustainability"]):
                potential_stakeholders = ["Environmental Scientist", "Policy Maker", "Sustainability Officer", "Concerned Citizen"]
            else:
                # Generic stakeholders for any domain
                potential_stakeholders = ["Professional", "Researcher", "Student", "Manager", "Consultant"]
        
        # Clean up and deduplicate stakeholders
        cleaned_stakeholders = []
        for stakeholder in potential_stakeholders:
            # Clean up the stakeholder string
            cleaned = stakeholder.strip().title()
            if cleaned and len(cleaned) > 3 and cleaned not in cleaned_stakeholders:
                cleaned_stakeholders.append(cleaned)
        
        # Ensure we have enough unique stakeholders
        if len(cleaned_stakeholders) < ctx.deps.num_scenarios:
            # Add generic stakeholders if needed
            generic_stakeholders = ["Professional", "Researcher", "Student", "Manager", "Consultant",
                                   "Specialist", "Analyst", "Director", "Coordinator", "Advisor"]
            for generic in generic_stakeholders:
                if generic not in cleaned_stakeholders:
                    cleaned_stakeholders.append(generic)
                if len(cleaned_stakeholders) >= ctx.deps.num_scenarios:
                    break
        
        # Extract potential goals and backgrounds from the content
        goal_phrases = []
        background_phrases = []
        
        # Look for goal-oriented phrases
        goal_patterns = [
            r'(?:aims|goals|objectives|purposes) (?:to|of|for) (.{10,60}?)(?:\.|\n)',
            r'(?:in order to|so as to|intended to) (.{10,60}?)(?:\.|\n)',
            r'(?:seeking|looking|aiming) (?:to|for) (.{10,60}?)(?:\.|\n)'
        ]
        
        for pattern in goal_patterns:
            matches = re.findall(pattern, combined_content)
            goal_phrases.extend(matches)
        
        # Look for background-oriented phrases
        background_patterns = [
            r'(?:background|experience|expertise) (?:in|with|of) (.{10,60}?)(?:\.|\n)',
            r'(?:trained|educated|specialized) (?:in|as|with) (.{10,60}?)(?:\.|\n)',
            r'(?:working|practicing|operating) (?:in|as|with) (.{10,60}?)(?:\.|\n)'
        ]
        
        for pattern in background_patterns:
            matches = re.findall(pattern, combined_content)
            background_phrases.extend(matches)
        
        # Clean up goal and background phrases
        cleaned_goals = [goal.strip() for goal in goal_phrases if len(goal.strip()) > 10]
        cleaned_backgrounds = [bg.strip() for bg in background_phrases if len(bg.strip()) > 10]
        
        # If we couldn't extract specific goals or backgrounds, generate them based on domain and topics
        if not cleaned_goals:
            # Use a default topic if the list is empty
            default_topic = topics[0] if topics else domain
            cleaned_goals = [
                f"Understand the implications of {default_topic} in {domain}",
                f"Implement best practices for {default_topic}",
                f"Stay updated on developments in {domain}",
                f"Solve problems related to {default_topic}",
                f"Make informed decisions about {domain}"
            ]
        
        if not cleaned_backgrounds:
            # Use a default topic if the list is empty
            default_topic = topics[0] if topics else domain
            cleaned_backgrounds = [
                f"Works in the field of {domain}",
                f"Has experience with {default_topic}",
                f"Studies {domain} professionally",
                f"Manages projects related to {default_topic}",
                f"Advises organizations on {domain}"
            ]
        
        # Generate personas based on extracted stakeholders, goals, and backgrounds
        import random
        personas = []
        num_personas = min(ctx.deps.num_scenarios, len(cleaned_stakeholders))
        
        for i in range(num_personas):
            stakeholder = cleaned_stakeholders[i]
            
            # Select a random goal and background, or generate them based on the stakeholder and topics
            if cleaned_goals:
                goal = random.choice(cleaned_goals).replace("{topic}", random.choice(topics)).replace("{domain}", domain)
            else:
                goal = f"Understand and apply {random.choice(topics)} in their work"
                
            if cleaned_backgrounds:
                background = random.choice(cleaned_backgrounds).replace("{topic}", random.choice(topics)).replace("{domain}", domain)
            else:
                background = f"Works with {random.choice(topics)} in the {domain} field"
            
            # Create the persona
            persona = {
                "name": f"User {i+1}",
                "type": stakeholder,
                "background": background,
                "goals": goal
            }
            
            personas.append(persona)
        
        return personas
    
    @scenario_planning_agent.tool
    async def generate_initial_questions(ctx: RunContext[ScenarioPlanningDeps], domain: str, topics: List[str], personas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate initial questions for each persona based on document content, domain and topics.
        
        Args:
            ctx: The run context containing document chunks
            domain: The identified domain
            topics: List of key topics
            personas: List of user personas
            
        Returns:
            List of scenarios with initial questions
        """
        # Get a representative sample of document chunks for analysis
        sample_size = min(10, len(ctx.deps.document_chunks))
        sample_chunks = ctx.deps.document_chunks[:sample_size]
        
        # Combine content from all sample chunks for analysis
        combined_content = ""
        for chunk in sample_chunks:
            content = chunk.get("content", "")
            if content:
                combined_content += content + "\n\n"
        
        # Extract potential questions from the content
        import re
        
        # Look for question-like patterns in the content
        question_patterns = [
            r'\b[Ww]hat (?:is|are) .{10,100}?\?',
            r'\b[Hh]ow (?:to|do|does|can) .{10,100}?\?',
            r'\b[Ww]hy (?:is|are|do|does) .{10,100}?\?',
            r'\b[Ww]hen (?:should|can|will) .{10,100}?\?',
            r'\b[Ww]hich .{10,100}?\?'
        ]
        
        potential_questions = []
        for pattern in question_patterns:
            matches = re.findall(pattern, combined_content)
            potential_questions.extend(matches)
        
        # Clean up extracted questions
        cleaned_questions = []
        for question in potential_questions:
            # Clean up the question
            cleaned = question.strip()
            if cleaned and len(cleaned) > 15 and cleaned not in cleaned_questions:
                cleaned_questions.append(cleaned)
        
        # Generate scenarios for each persona
        scenarios = []
        
        for i, persona in enumerate(personas):
            # Select a topic focus for this scenario
            import random
            topic_focus = random.choice(topics) if topics else domain
            
            # Generate context based on persona and topic
            context = f"{persona['type']} seeking information about {topic_focus}"
            
            # Generate information needs based on persona goals
            information_needs = []
            
            # Extract potential information needs from persona goals
            goals = persona.get("goals", "")
            if "understand" in goals.lower():
                information_needs.append(f"Understanding {topic_focus}")
            if "implement" in goals.lower() or "apply" in goals.lower():
                information_needs.append(f"Implementation strategies")
            if "analyze" in goals.lower() or "assess" in goals.lower():
                information_needs.append(f"Analysis methods")
            
            # Add some generic information needs if we don't have enough
            if len(information_needs) < 3:
                generic_needs = [
                    f"Key concepts in {topic_focus}",
                    f"Recent developments in {topic_focus}",
                    f"Best practices for {topic_focus}",
                    f"Challenges related to {topic_focus}",
                    f"Future trends in {topic_focus}"
                ]
                # Add generic needs until we have at least 3
                while len(information_needs) < 3 and generic_needs:
                    need = generic_needs.pop(0)
                    if need not in information_needs:
                        information_needs.append(need)
            
            # Generate an initial question
            initial_question = ""
            
            # If we have extracted questions from the content, try to use one that's relevant to the persona and topic
            if cleaned_questions:
                # Try to find a question that matches the persona's interests
                relevant_questions = []
                for question in cleaned_questions:
                    # Check if the question contains the topic or is related to the persona's goals
                    if (topic_focus.lower() in question.lower() or
                        any(goal_word in question.lower() for goal_word in goals.lower().split())):
                        relevant_questions.append(question)
                
                if relevant_questions:
                    initial_question = random.choice(relevant_questions)
            
            # If we couldn't find a relevant extracted question, generate one based on persona type and goals
            if not initial_question:
                question_templates = [
                    "What are the key aspects of {topic} that I should know about as a {persona_type}?",
                    "How can {topic} help me achieve my goal to {goal}?",
                    "What are the latest developments in {topic} relevant to {persona_type}s?",
                    "How does {topic} impact {persona_background}?",
                    "What strategies should I consider for {topic} given my background in {persona_background}?"
                ]
                
                # Select a random template and fill it
                question_template = random.choice(question_templates)
                initial_question = question_template.format(
                    topic=topic_focus,
                    persona_type=persona.get("type", "professional"),
                    goal=persona.get("goals", "improve understanding").split(" ")[-1],
                    persona_background=persona.get("background", "the field").split(" ")[-1]
                )
            
            # Create the scenario
            scenario = {
                "scenario_id": i + 1,
                "title": f"{persona['type']} Inquiry about {topic_focus}",
                "persona": {
                    "name": persona.get("name", f"User {i+1}"),
                    "type": persona.get("type", "User"),
                    "background": persona.get("background", "General background"),
                    "goals": persona.get("goals", "Learn more")
                },
                "context": context,
                "initial_question": initial_question,
                "information_needs": information_needs
            }
            
            scenarios.append(scenario)
        
        return scenarios
    
    return scenario_planning_agent