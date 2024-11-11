from langgraph.graph import StateGraph, END
from typing import Dict, TypedDict, List, Tuple, Union
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
import json
import re
import traceback
from prompts import (
    AIDA,
    PAS,
    BAB,
    FOURPs,
    TAS,
    FAB,
    SCQA,
    FOURCs,
    QUEST,
    SCH,
    CLARITY,
    STORYTELLING,
    CREATIVITY,
    AUTHENTICITY,
    IMPACT
)
import asyncio

# Define types for the workflow
class WorkflowState(TypedDict):
    content_idea: str
    target_audience: str
    age: str
    format: str
    goal: str
    selected_formulas: List[str]
    formula_reasoning: Dict[str, str]  # Added to store reasoning for each formula
    drafts: Dict[str, str]
    scores: Dict[str, Dict[str, Union[Dict[str, float], float]]]
    feedback: Dict[str, str]
    final_summary: Dict[str, str]
    revision_count: int

# Define available agents
AVAILABLE_FORMULAS = [AIDA, PAS, BAB, FOURPs,
                      TAS, FAB, SCQA, FOURCs, QUEST, SCH]

SCORING_CRITERIA = [CLARITY, STORYTELLING, CREATIVITY, AUTHENTICITY, IMPACT]

class TaskAgent:
    def __init__(self, model):
        self.model = model

    async def _parse_formula_selection(self, response_text: str) -> Tuple[List[str], Dict[str, str]]:
        """Parse the LLM response to extract selected formulas and their reasoning."""
        try:
            prompt = PromptTemplate(
                template="Extract the formulas and their reasoning from this text and return ONLY a JSON object like this: {{'selected_formulas': ['formula1', 'formula2'], 'reasoning': {{'formula1': 'reason1', 'formula2': 'reason2'}}}}\n\nText: {text}",
                input_variables=["text"]
            )

            formatted_prompt = prompt.format(text=response_text)

            try:
                model_response = self.model.invoke([
                    HumanMessage(content=formatted_prompt)
                ])
                
                try:
                    json_str = model_response.content.strip()
                    if "```json" in json_str:
                        json_str = json_str.split("```json")[1].split("```")[0].strip()
                    
                    json_output = json.loads(json_str)
                    selected_formulas = json_output.get("selected_formulas", [])
                    reasoning = json_output.get("reasoning", {})
                    return selected_formulas, reasoning
                    
                except json.JSONDecodeError as je:
                    print("JSON Parsing Error:")
                    print(traceback.format_exc())
                    return [AIDA, PAS], {}
                    
            except Exception as me:
                print("Model Invocation Error:")
                print(traceback.format_exc())
                return [AIDA, PAS], {}
               
        except Exception as e:
            print("Overall Error:")
            print(traceback.format_exc())
            return [AIDA, PAS], {}

    # Task Agent for selecting appropriate copywriting agents
    async def task_agent(self, state: WorkflowState) -> Dict:
        """Select appropriate copywriting formulas based on project requirements."""
        prompt = f"""
        As a copywriting expert, analyze the following project requirements:

        Given the following requirements:
        - Content idea: {state['content_idea']}
        - Target audience: {state['target_audience']}
        - Age: {state['age']}
        - Format: {state['format']}
        - Goal: {state['goal']}

        Select 1-3 most suitable formulas from: {', '.join(AVAILABLE_FORMULAS)}
        Explain your reasoning for each selection on how it algns with: {state['target_audience']}, {state['age']}, {state['format']}, {state['goal']}.
        Format your response as a structured analysis for each selected formula.
        """

        response = await self.model.ainvoke([HumanMessage(content=prompt)])
        selected_formulas, reasoning = await self._parse_formula_selection(
            response.content)

        return {
            "selected_formulas": selected_formulas,
            "formula_reasoning": reasoning,  # Added to include reasoning in the state
            "revision_count": 0
        }

class GenerateCopy:
    def __init__(self, model):
        self.model = model

    # Copywriting Agents
    async def generate_copy(self, state: WorkflowState) -> Dict:
        drafts = {}

        for formula in state["selected_formulas"]:
            prompt = f"""
            You are a professional copywriter specializing in the {formula} formula.
            Create compelling copy for the following project:

            Content Idea: {state['content_idea']}
            Target Audience: {state['target_audience']}
            Age Range: {state['age']}
            Content Format: {state['format']}
            Marketing Goal: {state['goal']}

            Requirements:
            1. Strictly follow the {formula} framework structure
            2. Maintain a consistent tone aligned with the target audience
            3. Ensure the copy length is appropriate for the specified format
            4. Include a clear call-to-action aligned with the marketing goal
            5. You will be valuated based on the following criteria, and please try to get the highest score:
            . {CLARITY} (1-10): Evaluate message clarity and readability
            . {STORYTELLING} (1-10): Assess narrative flow and engagement
            . {CREATIVITY} (1-10): Rate originality and innovative approach
            . {AUTHENTICITY} (1-10): Measure genuineness and brand alignment
            . {IMPACT} (1-10): Evaluate persuasiveness and call-to-action effectiveness

            Generate the copy and briefly explain how each part aligns with the {formula} framework.
            """

            response = await self.model.ainvoke([HumanMessage(content=prompt)])
            drafts[formula] = response.content

        # Increment revision count when generating new copy
        return {
            "drafts": drafts,
            "revision_count": state.get("revision_count", 0) + 1
        }

class ScoringAgent:
    def __init__(self, model):
        self.model = model

    def _fix_json_format(self, text: str) -> str:
        """Fix common JSON formatting issues."""
        # Remove any markdown formatting
        if "```" in text:
            pattern = r"```(?:json)?(.*?)```"
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                text = matches[0]
        
        # Find the JSON object
        json_pattern = r"\{.*\}"
        matches = re.findall(json_pattern, text, re.DOTALL)
        if matches:
            text = matches[0]

        # Remove whitespace and newlines
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Add missing commas between properties
        text = re.sub(r'(\d+|\w+"|"[^"]+\")\s+("(?:[^"]|\\")*":)', r'\1,\2', text)
        text = re.sub(r'(\})\s+("(?:[^"]|\\")*":)', r'\1,\2', text)
        
        return text

    def _extract_scores(self, text: str) -> Dict:
        """Extract scores from text using regex patterns."""
        try:
            criteria = {}
            # Look for patterns like "clarity": 8 or "clarity":8
            score_pattern = r'"(\w+)":\s*(\d+(?:\.\d+)?)'
            matches = re.findall(score_pattern, text)
            
            for criterion, score in matches:
                criterion = criterion.lower()
                if criterion in [c.lower() for c in SCORING_CRITERIA]:
                    criteria[criterion] = float(score)
            
            # Extract average score
            avg_pattern = r'"average":\s*(\d+(?:\.\d+)?)'
            avg_match = re.search(avg_pattern, text)
            average = float(avg_match.group(1)) if avg_match else sum(criteria.values()) / len(criteria)
            
            # Extract feedback
            feedback_pattern = r'"feedback":\s*"([^"]+)"'
            feedback_match = re.search(feedback_pattern, text)
            feedback = feedback_match.group(1) if feedback_match else "No feedback available"
            
            return {
                "criteria": criteria,
                "average": average,
                "feedback": feedback
            }
        except Exception as e:
            print(f"Error extracting scores: {str(e)}")
            raise

    async def _parse_scores(self, response_text: str) -> Dict:
        """Parse the scoring response to extract scores and feedback."""
        try:
            # First try to clean and parse as JSON
            cleaned_json = self._fix_json_format(response_text)
            try:
                result = json.loads(cleaned_json)
                # Validate the structure
                if "criteria" in result and "average" in result and "feedback" in result:
                    return result
            except json.JSONDecodeError:
                # If JSON parsing fails, try regex extraction
                result = self._extract_scores(cleaned_json)
                if result["criteria"] and len(result["criteria"]) == len(SCORING_CRITERIA):
                    return result
            
            # If both methods fail, try one more time with the model
            prompt = """
            Parse the following text and return ONLY a valid JSON object with scores and feedback.
            Format must be exactly:
            {
                "criteria": {
                    "clarity": 7,
                    "storytelling": 7,
                    "creativity": 7,
                    "authenticity": 7,
                    "impact": 7
                },
                "average": 7,
                "feedback": "feedback text"
            }
            
            Text to parse:
            {text}
            """
            
            model_response = await self.model.ainvoke([
                HumanMessage(content=prompt.format(text=response_text))
            ])
            
            cleaned_json = self._fix_json_format(model_response.content)
            result = json.loads(cleaned_json)
            
            # Final validation
            if not all(key in result for key in ["criteria", "average", "feedback"]):
                raise ValueError("Missing required keys in parsed result")
            
            return result
            
        except Exception as e:
            print(f"Error parsing scores: {str(e)}")
            print(f"Response text: {response_text[:200]}...")
            print(traceback.format_exc())
            
            # Return default scores as fallback
            return {
                "criteria": {
                    "clarity": 7.0,
                    "storytelling": 7.0,
                    "creativity": 7.0,
                    "authenticity": 7.0,
                    "impact": 7.0
                },
                "average": 7.0,
                "feedback": "Error parsing feedback. Using default scores."
            }

    async def scoring_agent(self, state: WorkflowState) -> Dict:
        scores = {}
        feedback = {}

        for formula, draft in state["drafts"].items():
            prompt = f"""
            Evaluate this copy and return ONLY a JSON object with scores and feedback.
            Format must be exactly as shown - include all commas, no extra text:
            {{
                "criteria": {{
                    "clarity": 7,
                    "storytelling": 7,
                    "creativity": 7,
                    "authenticity": 7,
                    "impact": 7
                }},
                "average": 7,
                "feedback": "feedback text"
            }}

            Criteria:
            1. {CLARITY} (1-10): Evaluate message clarity and readability
            2. {STORYTELLING} (1-10): Assess narrative flow and engagement
            3. {CREATIVITY} (1-10): Rate originality and innovative approach
            4. {AUTHENTICITY} (1-10): Measure genuineness and brand alignment
            5. {IMPACT} (1-10): Evaluate persuasiveness and call-to-action effectiveness

            Copy to evaluate:
            {draft}

            Context:
            - Target Audience: {state['target_audience']}
            - Age Range: {state['age']}
            - Goal: {state['goal']}
            - Format: {state['format']}
            """

            response = await self.model.ainvoke([HumanMessage(content=prompt)])
            parsed_response = await self._parse_scores(response.content)

            scores[formula] = {
                "criteria": parsed_response["criteria"],
                "average": parsed_response["average"]
            }
            feedback[formula] = parsed_response["feedback"]

        return {
            "scores": scores,            
            "feedback": feedback
        }

def should_revise(state: WorkflowState) -> bool:
    """Determine if any formula needs revision based on average scores and revision count."""
    # Get current revision count, default to 0 if not set
    revision_count = state.get("revision_count", 0)
    
    # Check if we've reached the maximum number of revisions (3)
    if revision_count >= 3:
        return False
    
    # Check if any score is below threshold
    needs_revision = any(
        score_data["average"] < 8.0
        for score_data in state["scores"].values()
    )
    
    return needs_revision

class CreateSummary:
    def __init__(self, model):
        self.model = model

    # Summary node
    async def create_summary(self, state: WorkflowState) -> Dict:
        # Prepare detailed summary
        summary = {
            "selected_formulas": state["selected_formulas"],
            "formula_reasoning": state["formula_reasoning"],  # Added to include reasoning in summary
            "drafts": state["drafts"],
            "scores": state["scores"],
            "feedback": state["feedback"],
            "best_performing": self._get_best_performing(state["scores"]),
            "improvement_suggestions": self._get_improvement_suggestions(state)
        }

        return {"final_summary": summary}

    def _get_best_performing(self, scores: Dict[str, Dict[str, float]]) -> str:
        """Identify the best performing formula based on average scores."""
        return max(
            scores.items(),
            key=lambda x: x[1]["average"]
        )[0]

    def _get_improvement_suggestions(self, state: WorkflowState) -> Dict[str, List[str]]:
        """Generate specific improvement suggestions for each formula."""
        suggestions = {}
        for formula, score_data in state["scores"].items():
            formula_suggestions = []
            for criterion, score in score_data["criteria"].items():
                if score < 8.0:
                    formula_suggestions.append(
                        f"Improve {criterion}: Current score {score}")
            suggestions[formula] = formula_suggestions
        return suggestions

# Define the workflow graph
def create_copywriting_workflow(model) -> StateGraph:
    # Create workflow graph
    workflow = StateGraph(WorkflowState)

    task_agent = TaskAgent(model)
    generate_copy = GenerateCopy(model)
    scoring_agent = ScoringAgent(model)
    create_summary = CreateSummary(model)

    # Add nodes
    workflow.add_node("task_agent", task_agent.task_agent)
    workflow.add_node("generate_copy", generate_copy.generate_copy)
    workflow.add_node("scoring_agent", scoring_agent.scoring_agent)
    workflow.add_node("create_summary", create_summary.create_summary)

    # Define edges
    workflow.set_entry_point("task_agent")
    workflow.add_edge("task_agent", "generate_copy")
    workflow.add_edge("generate_copy", "scoring_agent")

    # Add conditional routing
    workflow.add_conditional_edges(
        "scoring_agent",
        should_revise,
        {
            True: "generate_copy",  # If score <= 8.0 and revisions < 3, go back to generate copy
            False: "create_summary"  # If score > 8.0 or revisions >= 3, proceed to summary
        }
    )

    workflow.add_edge("create_summary", END)

    compiled_workflow = workflow.compile()
    return compiled_workflow
