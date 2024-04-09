# gpt-pilot - GPT Pilot doesn't just generate code, it builds apps!

# https://github.com/Pythagora-io/gpt-pilot


"""
How to start using gpt-pilot?
If you are using VS Code as your IDE, the easiest way to start is by downloading GPT Pilot VS Code extension.

Otherwise, you can use the CLI tool.

After you have Python and (optionally) PostgreSQL installed, follow these steps:

git clone https://github.com/Pythagora-io/gpt-pilot.git (clone the repo)
cd gpt-pilot
python -m venv pilot-env (create a virtual environment)
source pilot-env/bin/activate (or on Windows pilot-env\Scripts\activate) (activate the virtual environment)
pip install -r requirements.txt (install the dependencies)
cd pilot
mv .env.example .env (or on Windows copy .env.example .env) (create the .env file)
Add your environment to the .env file:
LLM Provider (OpenAI/Azure/Openrouter)
Your API key
database settings: SQLite/PostgreSQL (to change from SQLite to PostgreSQL, just set DATABASE_TYPE=postgres)
optionally set IGNORE_PATHS for the folders which shouldn't be tracked by GPT Pilot in workspace, useful to ignore folders created by compilers (i.e. IGNORE_PATHS=folder1,folder2,folder3)
python main.py (start GPT Pilot)
After, this, you can just follow the instructions in the terminal.

All generated code will be stored in the folder workspace inside the folder named after the app name you enter upon starting the pilot.
"""
