import streamlit as st
from langchain_groq import ChatGroq
from main import create_copywriting_workflow
from typing import Dict, Any, Optional
from langchain.callbacks.base import BaseCallbackHandler
import os
import asyncio

class StreamlitCallbackHandler(BaseCallbackHandler):
    """Custom callback handler to update Streamlit interface during processing"""

    def __init__(self, status_placeholder):
        self.status_placeholder = status_placeholder
        self.token_buffer = ""  # Buffer for token accumulation

    def on_llm_start(self, *args, **kwargs):
        self.status_placeholder.markdown("🤔 Thinking...")

    def on_llm_end(self, *args, **kwargs):
        self.status_placeholder.markdown("✅ Done!")

    def on_llm_new_token(self, token: str, **kwargs):
        # Accumulate tokens in buffer and update periodically
        self.token_buffer += token
        if len(self.token_buffer) > 10:  # Update every 10 tokens
            self.status_placeholder.markdown(f"Generating: {self.token_buffer}")
            self.token_buffer = ""

def setup_environment():
    """Set up environment variables from Streamlit secrets"""
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]["API_KEY"]
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_PROJECT"] = "Copywriter Master"

def setup_page():
    """Configure the Streamlit page settings"""
    st.set_page_config(
        page_title="Copywriter",
        page_icon="🎯",
        layout="wide"
    )
    # Add custom CSS
    st.markdown("""
        <style>
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size: 1.2rem;
        }
        .success-metric {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #f0f2f6;
        }
        </style>
    """, unsafe_allow_html=True)

def create_sidebar():
    """Create and configure the sidebar"""
    st.sidebar.header("🎯 Copywriting Master")
    st.sidebar.markdown(
        "This app recommends content copywriting tailored to match your social media brand positioning. "
        "To use this App, you need to provide a Groq API key, which you can get [here](https://console.groq.com/keys) for free."
    )

    st.sidebar.write("### Instructions")
    instructions = [
        "1️⃣ Enter your content idea/topic",
        "2️⃣ Define your target audience",
        "3️⃣ Choose content format",
        "4️⃣ Choose your goal",
        "5️⃣ Click 'Generate' to get sample copywriting"
    ]
    for instruction in instructions:
        st.sidebar.write(instruction)

    try:
        st.sidebar.image("assets/logo01.jpg", use_container_width=True)
    except FileNotFoundError:
        st.sidebar.warning("Logo file not found. Please check the assets directory.")

def get_api_key() -> Optional[str]:
    """Get and validate the Groq API key"""
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="Enter your Groq API key...",
        help="Your key will not be stored"
    )
    if not api_key:
        st.info("Please add your Groq API key to continue.", icon="🔑")
    return api_key

def create_input_form() -> Optional[Dict[str, str]]:
    """Create and handle the input form"""
    with st.form("positioning_form"):
        col1, col2 = st.columns(2)

        with col1:
            content_idea = st.text_area(
                "Content Idea",
                placeholder="e.g., No-code AI tools can significantly improve small business productivity.",
                help="What is the content idea or topic on your mind? Please be specific!"
            )

        with col2:
            target_audience = st.text_area(
                "Target Audience",
                placeholder="e.g., Small business owners with limited technical knowledge",
                help="Who are you trying to reach and serve?"
            )

        age = st.radio("Select the target age of your audience:", ["18-24", "25-34", "35-44", "45-54", "55+"], horizontal=True)
        format = st.radio("Select your content format:", ["Short video script", "Social Media Post", "LinkedIn Post", "Case studies", "Marketing Email"], horizontal=True)
        goal = st.radio("Select the goal of your content:", ["Awareness", "Engagement", "Education", "Conversion"], horizontal=True)

        submit_button = st.form_submit_button("Generate")

        if submit_button:
            if not all([content_idea, target_audience, age, format, goal]):
                st.error("Please fill in all fields before generating sample copywriting.")
                return None

            return {
                "content_idea": content_idea,
                "target_audience": target_audience,
                "age": age,
                "format": format,
                "goal": goal,
            }
    return None

def display_results(workflow_state: Dict[str, Any]):
    """Display the generated content and analysis in tabs"""
    if not workflow_state.get("drafts"):
        st.write("No results to display.")
        return

    tabs = st.tabs(["Generated Copy", "Performance Analysis", "Formula Selection", "Summary"])

    with tabs[0]:
        st.subheader("📝 Generated Copy Variations")
        for formula, draft in workflow_state.get("drafts", {}).items():
            with st.expander(f"{formula} Formula Version"):
                st.markdown(draft)
                scores = workflow_state.get("scores", {}).get(formula, {})
                cols = st.columns(len(scores.get("criteria", [])))
                for col, (criterion, score) in zip(cols, scores["criteria"].items()):
                    with col:
                        st.metric(label=criterion, value=f"{score:.1f}/10", delta=f"{score - 7.5:.1f}")

    with tabs[1]:
        st.subheader("📊 Performance Analysis")
        for formula, score_data in workflow_state.get("scores", {}).items():
            st.markdown(f"### {formula} Formula Analysis")
            st.metric(label="Overall Score", value=f"{score_data['average']:.1f}/10", delta=f"{score_data['average'] - 8.5:.1f}")

    with tabs[2]:
        st.subheader("🎯 Formula Selection Rationale")
        for formula in workflow_state.get("selected_formulas", []):
            with st.expander(f"Why {formula}?"):
                st.markdown(workflow_state.get("formula_reasoning", {}).get(formula, "No reasoning available."))

    with tabs[3]:
        st.subheader("📋 Executive Summary")
        summary = workflow_state.get("final_summary", {})
        st.markdown(f"### 🏆 Best Performing Version: {summary.get('best_performing', 'N/A')}")
        for formula, suggestions in summary.get("improvement_suggestions", {}).items():
            with st.expander(f"Suggestions for {formula}"):
                for suggestion in suggestions:
                    st.markdown(f"- {suggestion}")

async def run_workflow_async(state_machine, workflow_state):
    try:
        final_state = await state_machine.ainvoke(input=workflow_state)
        return final_state
    except Exception as e:
        st.error(f"An error occurred during workflow execution: {e}")
        st.exception(e)
        return None
    
def initialize_workflow(api_key):
    try:
        model = ChatGroq(temperature=0.3, groq_api_key=api_key, model="llama-3.1-70b-versatile", request_timeout=60) # Or your LLM
        workflow = create_copywriting_workflow(model)
        return workflow
    except Exception as e:  # Handle any exceptions during workflow initialization
        st.error(f"An error occurred during workflow initialization: {e}")
        st.exception(e)
        return None

def main():
    try:
        setup_environment()
        setup_page()
        create_sidebar()

        st.title("🎯 Copywriting Master")

        api_key = get_api_key()
        if not api_key:
            return

        workflow = initialize_workflow(api_key)
        if not workflow:
            return  # Exit if workflow initialization failed

        status_placeholder = st.empty()
        callback_handler = StreamlitCallbackHandler(status_placeholder)

        input_data = create_input_form()

        if input_data:
            with st.spinner("Generating optimized copy..."):
                initial_workflow_state = {  # Initialize the workflow state
                    **input_data,
                    "selected_formulas": [],
                    "drafts": {},
                    "scores": {},
                    "feedback": {},
                    "final_summary": {}
                }

                final_state = asyncio.run(
                    run_workflow_async(workflow, initial_workflow_state),)
                

                if final_state: # Check if the workflow completed successfully
                    display_results(final_state)

    except Exception as e:
        st.error("An unexpected error occurred. Please try again.")
        st.exception(e)


if __name__ == "__main__":
    main()
