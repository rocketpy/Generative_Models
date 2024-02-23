import os
from openai import OpenAI


client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"]
)

ticket_text = "Some text here ..."
some_text = f"Text: {ticket_text}"

response = client.chat.completions.create(model="gpt-3.5-turbo",
                                          messages=[{"role": "user", "content": some_text}],
                                          max_tokens=100)
# print("Response: \n"+response.choices[0].message.content)


# To save data sets
import json
import weave
from weave.weaveflow import Dataset


weave.init('test')
with open('test.json') as f:
    data = json.load(f)
    dataset = Dataset(data)
    weave.publish(dataset, 'data')
