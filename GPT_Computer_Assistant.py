# GPT Computer Assistant - this is an alternative work for providing ChatGPT MacOS app to Windows and Linux. 

# https://github.com/onuratakan/gpt-computer-assistant

"""
Installation && Run
Needed >= Python 3.9

pip3 install gpt-computer-assistant
computerassistant
"""

# Agent Infrastructure | NEW
# To create crewai agents and using it into gpt-computer-assistant gui and tools.

# pip3 install gpt-computer-assistant[agentic]


from gpt_computer_assistant import Agent, start

manager = Agent(
  role='Project Manager',
  goal='understands project needs and assist coder',
  backstory="""You're a manager at a large company.""",
)

coder = Agent(
  role='Senior Python Coder',
  goal='writing python scripts and copying to clipboard',
  backstory="""You're a python developer at a large company.""",
)

start()


# Adding Custom Tools | NEW
# Now you are able to add custom tools that run in the agentic infra and assistant processes.

from gpt_computer_assistant import Tool, start

@Tool
def sum_tool(first_number: int, second_number: int) -> str:
    """Useful for when you need to sum two numbers together."""
    return first_number + second_number

start()
