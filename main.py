import os
from dotenv import load_dotenv, find_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
import openai 
import google.api_core.exceptions 
import tiktoken 

# --- Environment and Page Config ---
_ = load_dotenv(find_dotenv())

st.set_page_config(
    page_title="Chat Bot",
    page_icon="🤖"
)

# --- Load CSS ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

# --- Pricing Data (USD per 1k tokens) ---
openai_pricing = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},  
    "gpt-4o": {"input": 0.0025, "output": 0.01},        
}

# --- Token Counting Function (OpenAI Only) ---
def count_openai_tokens(text, model_name="gpt-4o-mini"): 
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base") 
    return len(encoding.encode(text))

# --- Session State Initialization ---
# API Keys
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = os.environ.get("OPENAI_API_KEY")
if 'google_api_key' not in st.session_state:
    st.session_state.google_api_key = os.environ.get("GOOGLE_API_KEY")

# Model Provider Choice
if 'selected_provider' not in st.session_state:
    st.session_state.selected_provider = "OpenAI"

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# LangChain Memory
if 'lc_memory' not in st.session_state:
    st.session_state.lc_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Cost Tracking (OpenAI only)
if 'total_cost' not in st.session_state:
    st.session_state.total_cost = 0.0

# --- Sidebar ---
st.sidebar.title("Configuration")

# Provider Selection
st.session_state.selected_provider = st.sidebar.radio(
    "Choose LLM Provider:",
    ("OpenAI", "Google", "OSS LLM"),
    key="provider_radio",
)

provider = st.session_state.selected_provider

# --- API Key Handling (Conditional) ---
api_key_needed = False
api_key_provided = False

if provider == "OpenAI":
    if not st.session_state.openai_api_key:
        api_key_needed = True
        with st.expander("Enter OpenAI API Key", expanded=True):
            api_key_input = st.text_input(
                "OpenAI API Key:", type="password", key="openai_api_widget"
            )
            if api_key_input:
                if api_key_input.startswith("sk-") and len(api_key_input) > 50:
                    st.session_state.openai_api_key = api_key_input
                    st.success("OpenAI Key Accepted!", icon="✅")
                    st.rerun()
                else:
                    st.error("Invalid OpenAI API Key format.")
    else:
        api_key_provided = True
        masked_key = st.session_state.openai_api_key[:5] + "..." + st.session_state.openai_api_key[-4:]
        st.sidebar.success(f"Using OpenAI Key: {masked_key}", icon="🔑")

elif provider == "Google":
    if not st.session_state.google_api_key:
        api_key_needed = True
        with st.expander("Enter Google API Key", expanded=True):
            api_key_input = st.text_input(
                "Google API Key (Gemini):", type="password", key="google_api_widget"
            )
            if api_key_input and len(api_key_input) > 30:
                st.session_state.google_api_key = api_key_input
                st.success("Google Key Accepted!", icon="✅")
                st.rerun()
            elif api_key_input:
                 st.warning("Key entered, ensure it is correct.")
    else:
        api_key_provided = True
        masked_key = st.session_state.google_api_key[:5] + "..." + st.session_state.google_api_key[-4:]
        st.sidebar.success(f"Using Google Key: {masked_key}", icon="🔑")

elif provider == "OSS LLM":
    api_key_provided = True
    st.sidebar.info("Using OSS LLM", icon="🏠")

if api_key_needed and not api_key_provided and provider != "OSS LLM":
    st.warning(f"Please provide the API key for {provider} to continue.")
    st.stop()

# --- Model & Parameter Selection (Conditional) ---

llm = None
model_name = None

if provider == "OpenAI":
    model_name = st.sidebar.selectbox(
        "Choose OpenAI Model", options=["gpt-4o-mini", "gpt-4o-latest"], index=0, key="openai_model_select"
    )
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.01, key="openai_temp")
    top_p = st.sidebar.slider("Top P", 0.0, 1.0, 0.5, 0.01, key="openai_top_p")
    max_tokens = st.sidebar.slider("Max Tokens", 50, 4000, 300, 10, key="openai_max_tokens")

    try:
        llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=st.session_state.openai_api_key,
            model_kwargs={ "top_p": top_p }
        )
        st.sidebar.caption(f"Initialized OpenAI: {model_name}")
    except openai.AuthenticationError:
        st.error("OpenAI Authentication Error. Check your API key.")
        st.session_state.openai_api_key = None
        st.stop()
    except Exception as e:
        st.error(f"Error initializing OpenAI LLM: {e}")
        st.stop()

elif provider == "Google":
    model_name = st.sidebar.selectbox(
        "Choose Google Model", options=["gemini-1.5-flash", "gemini-2.0-flash-thinking-exp-01-21", "gemini-2.5-pro-exp-03-25"], index=0, key="google_model_select"
    )
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.3, 0.01, key="google_temp")
    top_p = st.sidebar.slider("Top P", 0.0, 1.0, 0.9, 0.01, key="google_top_p")
    max_tokens = st.sidebar.slider("Max Output Tokens", 50, 8192, 500, 10, key="google_max_tokens")

    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=st.session_state.google_api_key,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
            convert_system_message_to_human=True
        )
        st.sidebar.caption(f"Initialized Google: {model_name}")
    except (google.api_core.exceptions.PermissionDenied, google.api_core.exceptions.Unauthenticated) as e:
        st.error(f"Google Authentication Error ({type(e).__name__}). Check API key & Gemini API enabled.")
        st.session_state.google_api_key = None
        st.stop()
    except Exception as e:
        st.error(f"Error initializing Google LLM: {e}")
        st.stop()

elif provider == "OSS LLM":
    model_name = "gemma-3b-it"
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.01, key="local_gemma_temp")
    top_p = st.sidebar.slider("Top P", 0.0, 1.0, 0.95, 0.01, key="local_gemma_top_p")
    max_tokens = st.sidebar.slider("Max Tokens", 50, 2000, 300, 10, key="local_gemma_max_tokens")

    try:
        llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            openai_api_base="https://85aa-213-197-152-65.ngrok-free.app/v1",
            openai_api_key="dummy_key"
        )
        st.sidebar.caption(f"Initialized OSS LLM")
    except Exception as e:
        st.error(f"Error initializing OSS LLM: {e}")
        st.stop()


# --- LangChain Conversation Setup (Common) ---
if llm:
    system_message_prompt = SystemMessagePromptTemplate.from_template("You're a pleasant and helpful chatbot.")
    memory_placeholder = MessagesPlaceholder(variable_name="chat_history")
    human_message_prompt = HumanMessagePromptTemplate.from_template("{input}")

    prompt = ChatPromptTemplate.from_messages([
        system_message_prompt,
        memory_placeholder,
        human_message_prompt
    ])

    conversation = ConversationChain(
        llm=llm,
        prompt=prompt,
        memory=st.session_state.lc_memory,
        verbose=False
    )
else:
    st.error("LLM could not be initialized. Please check configuration.")
    st.stop()


# --- Display Existing Chat History ---
if not st.session_state.messages:
     with st.chat_message("assistant", avatar="🤖"):
          st.markdown("Hi there! Choose a provider and model in the sidebar, then ask me anything.")

for message in st.session_state.messages:
    avatar = "🤖" if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


# --- Chat Input and Processing ---
if user_input := st.chat_input("You:", key="chat_input", max_chars=300):

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    try:
        with st.spinner(f"Thinking with {provider}..."):
            # --- Cost Calculation Start (OpenAI Only) ---
            if provider == "OpenAI":
                input_tokens = count_openai_tokens(user_input, model_name)
                input_cost = (input_tokens / 1000) * openai_pricing[model_name]["input"]

                response = conversation.predict(input=user_input)

                output_tokens = count_openai_tokens(response, model_name)
                output_cost = (output_tokens / 1000) * openai_pricing[model_name]["output"]

                turn_cost = input_cost + output_cost
                st.session_state.total_cost += turn_cost
            else: 
                response = conversation.predict(input=user_input)
                turn_cost = 0.0
            # --- Cost Calculation End ---


        st.session_state.messages.append({"role": "assistant", "content": response, "cost": turn_cost}) # Store cost per message
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(response)

    except (openai.AuthenticationError, google.api_core.exceptions.PermissionDenied, google.api_core.exceptions.Unauthenticated) as auth_err:
         st.error(f"{provider} Authentication Error during chat: {auth_err}")
         if provider == "OpenAI": st.session_state.openai_api_key = None
         else: st.session_state.google_api_key = None
         if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
             st.session_state.messages.pop()
         st.rerun()
    except Exception as e:
        st.error(f"An error occurred during conversation with {provider}: {e}")
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            st.session_state.messages.pop()

# --- Sidebar Display of Total Cost (OpenAI only) ---
if provider == "OpenAI": # <--- Conditional display
    st.sidebar.subheader("Cost Tracking (OpenAI)")
    st.sidebar.metric("Total OpenAI Cost", f"${st.session_state.total_cost:.6f}")
    st.sidebar.caption("Estimated cost for OpenAI usage in this session.")

# --- Clear History Button ---
def clear_chat_history():
    st.session_state.messages = []
    st.session_state.lc_memory.clear()
    st.session_state.total_cost = 0.0 # Reset total cost

st.sidebar.button("Clear Chat History", on_click=clear_chat_history)