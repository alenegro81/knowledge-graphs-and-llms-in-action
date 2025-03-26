from openai import OpenAI

class LLM_Model():
    def __init__(self, url='http://localhost:11434/v1', key="default"):
        self.client = OpenAI(
          base_url= url,
          api_key = key, # required, but unused for open libraries
        )

    def generate(self, messages):
        response = self.client.chat.completions.create(
            model="llama3.1:latest",
            messages=messages,
            temperature=0,
            max_tokens=4000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        # It assumes as response the ChatGPT API format
        return response.choices[0].message.content