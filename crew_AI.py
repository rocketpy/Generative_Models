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
  
    # Examples:
    # llm=ollama_llm # was defined above in the file
    # llm=OpenAI(model_name="gpt-3.5", temperature=0.7)
    # For the OpenAI model you would need to import
    # from langchain_openai import OpenAI
)
writer = Agent(
    role='Tech Content Strategist',
    goal='Craft compelling content on tech advancements',
    backstory="""You are a renowned Content Strategist, known for your insightful and engaging articles.
    You transform complex concepts into compelling narratives.""",
    verbose=True,
    allow_delegation=True,
    # (optional) llm=ollama_llm\
)

# Create tasks for your agents
task1 = Task(
  description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024.
  Identify key trends, breakthrough technologies, and potential industry impacts.
  Your final answer MUST be a full analysis report""",
  agent=researcher
)

task2 = Task(
  description="""Using the insights provided, develop an engaging blog
  post that highlights the most significant AI advancements.
  Your post should be informative yet accessible, catering to a tech-savvy audience.
  Make it sound cool, avoid complex words so it doesn't sound like AI.
  Your final answer MUST be the full blog post of at least 4 paragraphs.""",
  agent=writer
)

# Instantiate your crew with a sequential process
crew = Crew(
  agents=[researcher, writer],
  tasks=[task1, task2],
  verbose=2, # You can set it to 1 or 2 to different logging levels
)

# Get your crew to work!
result = crew.kickoff()

print("********")
print(result)


# Examples

# AI Crew for Stock Analysis
"""
CrewAI Framework
CrewAI is designed to facilitate the collaboration of role-playing AI agents. In this example,
these agents work together to give a complete stock analysis and investment recommendation

Running the Script
It uses GPT-4 by default so you should have access to that to run it.

Disclaimer: This will use gpt-4 unless you changed it not to, and by doing so it will cost you money.

Configure Environment: Copy ``.env.example` and set up the environment variables for Browseless, Serper, SEC-API and OpenAI
Install Dependencies: Run poetry install --no-root.
Execute the Script: Run python main.py and input your idea.
Details & Explanation
Running the Script: Execute `python main.py`` and input the company to be analyzed when prompted.
The script will leverage the CrewAI framework to analyze the company and generate a detailed report.
Key Components:
./main.py: Main script file.
./stock_analysis_tasks.py: Main file with the tasks prompts.
./stock_analysis_agents.py: Main file with the agents creation.
./tools: Contains tool classes used by the agents.
"""

# Using GPT 3.5
# CrewAI allow you to pass an llm argument to the agent construtor, that will be it's brain,
# so changing the agent to use GPT-3.5 instead of GPT-4 is as simple as passing that argument on the agent you want to use that LLM (in main.py).


from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model='gpt-3.5') # Loading GPT-3.5

def local_expert(self):
	return Agent(
      role='The Best Financial Analyst',
      goal="""Impress all customers with your financial data 
      and market trends analysis""",
      backstory="""The most seasoned financial analyst with 
      lots of expertise in stock market analysis and investment
      strategies that is working for a super important customer.""",
      verbose=True,
      llm=llm, # <----- passing our llm reference here
      tools=[
        BrowserTools.scrape_and_summarize_website,
        SearchTools.search_internet,
        CalculatorTools.calculate,
        SECTools.search_10q,
        SECTools.search_10k
      ]
    )
