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
