# Vilkyt Chatbot

A simple chatbot app deployed online at [vilkyt.streamlit.app](https://vilkyt.streamlit.app/).

## Available LLMs

Choose from three language model providers:

### 1. OpenAI
- **Models available:** `gpt-4o-mini`, `gpt-4o`
- **API Key Required:** You must provide a valid OpenAI API key when selecting these models. The key is **not stored** and will be cleared after the session.

### 2. Gemini (Google)
- **Models available:** `gemini-1.5-flash`, `gemini-2.0-flash-thinking-exp-01-21`, `gemini-2.5-pro-exp-03-25`
- **API Key Required:** A valid Gemini API key is required when selecting these models. The key is **not stored** and will be cleared after the session.

### 3. OSS LLM (Open Source)
- **Model used:** `Gemini3 1B`
- **Note:** This OSS LLM is deployed via LLMStudio and runs occasionally, particularly during testing phases.

## Configuration Options

Common configuration parameters available in the sidebar:

- **Temperature:** Controls randomness in responses (higher values mean more creative outputs).
- **Top P:** Controls the probability cutoff for token sampling.
- **Max Tokens:** Sets the maximum number of tokens in the chatbot's responses.

## Installation in Another Environment

To install dependencies from `requirements.txt` in another environment, run:

```bash
uv pip install -r requirements.txt
```

## Privacy and Security

- API keys are **not stored** or persisted; they will be cleared immediately after each session ends.

---

For feedback or issues, please raise a ticket or contact the project maintainer.