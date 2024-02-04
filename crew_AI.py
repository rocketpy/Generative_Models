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

# Define your agents with roles and goals
researcher = Agent(
  role='Senior Research Analyst',
  goal='Uncover cutting-edge developments in AI and data science',
  backstory="""You work at a leading tech think tank.
  Your expertise lies in identifying emerging trends.
  You have a knack for dissecting complex data and presenting actionable insights.""",
  verbose=True,
  allow_delegation=False,
  tools=[search_tool]
  # You can pass an optional llm attribute specifying what mode you wanna use.
  # It can be a local model through Ollama / LM Studio or a remote
  # model like OpenAI, Mistral, Antrophic or others (https://python.langchain.com/docs/integrations/llms/)
  #
  # Examples:
  # llm=ollama_llm # was defined above in the file
  # llm=OpenAI(model_name="gpt-3.5", temperature=0.7)
  # For the OpenAI model you would need to import
  # from langchain_openai import OpenAI
)
