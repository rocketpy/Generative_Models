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

# How to start gpt-pilot in docker?
"""
git clone https://github.com/Pythagora-io/gpt-pilot.git (clone the repo)
Update the docker-compose.yml environment variables, which can be done via docker compose config. If you wish to use a local model,
please go to https://localai.io/basics/getting_started/.
By default, GPT Pilot will read & write to ~/gpt-pilot-workspace on your machine, you can also edit this in docker-compose.yml
run docker compose build. this will build a gpt-pilot container for you.
run docker compose up.
access the web terminal on port 7681
python main.py (start GPT Pilot)
This will start two containers, one being a new image built by the Dockerfile and a Postgres database.
The new image also has ttyd installed so that you can easily interact with gpt-pilot. Node is also installed on the image and port 3000 is exposed.
"""
