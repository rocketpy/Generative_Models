# Ollama Deep Researcher - Ollama Deep Researcher is a fully local web research assistant that uses any LLM hosted

# https://github.com/langchain-ai/ollama-deep-researcher


# Quickstart
"""
Mac

Download the Ollama app for Mac - https://ollama.com/download

Pull a local LLM from Ollama. As an example:

ollama pull deepseek-r1:8b

Clone the repository:
git clone https://github.com/langchain-ai/ollama-deep-researcher.git
cd ollama-deep-researcher

Select a web search tool:
By default, it will use DuckDuckGo for web search, which does not require an API key. 
But you can also use Tavily or Perplexity by adding their API keys to the environment file:

cp .env.example .env


# Running with LangGraph Studio

Mac
(Recommended) Create a virtual environment:
python -m venv .venv
source .venv/bin/activate

Launch LangGraph server:
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev


# Windows

(Recommended) Create a virtual environment:
Install Python 3.11 (and add to PATH during installation).
Restart your terminal to ensure Python is available, then create and activate a virtual environment:
python -m venv .venv
.venv\Scripts\Activate.ps1
Launch LangGraph server:
# Install dependencies
pip install -e .
pip install -U "langgraph-cli[inmem]"            

# Start the LangGraph server
langgraph dev
"""
