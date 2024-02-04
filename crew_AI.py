# crewAI - Cutting-edge framework for orchestrating role-playing, autonomous AI agents. 

# https://github.com/joaomdmoura/crewAI

# Installation:
# pip install crewai

# The example below also uses duckduckgo, so also install that
# pip install duckduckgo-search

# Setting Up Your Crew
import os
from crewai import Agent, Task, Crew, Process

os.environ["OPENAI_API_KEY"] = "YOUR KEY"

# You can choose to use a local model through Ollama for example. See ./docs/how-to/llm-connections.md for more information.
# from langchain.llms import Ollama
# ollama_llm = Ollama(model="openhermes")

# Install duckduckgo-search for this example:
# !pip install -U duckduckgo-search

from langchain_community.tools import DuckDuckGoSearchRun
search_tool = DuckDuckGoSearchRun()
