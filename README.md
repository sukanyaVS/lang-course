# Lang Course

A small Python project for exploring LangChain, LangGraph, and LLM integrations with OpenAI and Anthropic models.

## Prerequisites

Before you start, make sure you have:

- Python 3.13 or newer
- uv installed
- API keys for OpenAI and Anthropic

## Project setup

1. Clone the repository and move into the project folder:

   ```bash
   git clone <repository-url>
   cd lang-course
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   uv sync
   ```

   This installs the project dependencies from `pyproject.toml`, including:

   - `langchain`
   - `langchain-anthropic`
   - `langchain-core`
   - `langchain-openai`
   - `langchain-openapi`
   - `langgraph`
   - `python-dotenv`

3. Create a `.env` file in the project root and add your API keys:

   ```env
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   ```

   The code loads environment variables with `python-dotenv`, so these values will be available at runtime.

## Run the project

You can run the example scripts directly with uv:

```bash
uv run python .\src\lang_course\main.py
uv run python .\src\lang_course\prompt.py
```

The project also exposes a console script entry point:

```bash
uv run lang-course
```

## Notes

- The examples in `src/lang_course` use `langchain`, `langchain-openai`, and `langchain-anthropic`.
- Ensure your API keys are valid before running the LLM examples.
- If you are using PowerShell on Windows, the commands above are the expected format.
