# Gorilla: Large Language Model Connected with Massive APIs


# https://github.com/ShishirPatil/gorilla

"""
# About
Gorilla enables LLMs to use tools by invoking APIs. Given a natural language query, 
Gorilla comes up with the semantically- and syntactically- correct API to invoke.

# Features
With Gorilla, we are the first to demonstrate how to use LLMs to invoke 1,600+ 
(and growing) API calls accurately while reducing hallucination. 
This repository contains inference code for running Gorilla finetuned models, 
evaluation code for reproducing results from our paper, and APIBench - the largest collection of APIs, 
curated and easy to be trained on!

Since our initial release, we've served ~500k requests and witnessed incredible adoption by developers worldwide. 
The project has expanded to include tools, evaluations, leaderboard, 
end-to-end finetuning recipes, infrastructure components.
"""

# Installation Options
"""
Gorilla CLI - Fastest way to get started
pip install gorilla-cli
gorilla generate 100 random characters into a file called test.txt

# To Run Gorilla Locally
git clone https://github.com/ShishirPatil/gorilla.git
cd gorilla/inference
"""


# Use OpenFunctions
import openai

openai.api_key = "EMPTY"
openai.api_base = "http://luigi.millennium.berkeley.edu:8000/v1"

# Define your functions
functions = [{
    "name": "get_current_weather",
    "description": "Get weather in a location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        },
        "required": ["location"]
    }
}]


# Make API call
completion = openai.ChatCompletion.create(
    model="gorilla-openfunctions-v2",
    messages=[{"role": "user", "content": "What's the weather in San Francisco?"}],
    functions=functions
)
