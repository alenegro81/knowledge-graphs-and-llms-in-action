import os
from openai import OpenAI
from dotenv import load_dotenv

_ = load_dotenv()

class Agent:
    def __init__(self, model: str = "gpt-4o-mini", system: str = None):
        self.model = model
        self.system = system
        self.messages = list()
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

        if self.system is None or len(self.system) == 0:
            self.system = "You are an AI assistant providing straightforward concise answers."
        self.messages.append({"role": "system", "content": self.system})

    def __call__(self, message: str) -> str:
        self.messages.append({"role": "user", "content": message})
        answer = self.execute()
        self.messages.append({"role": "assistant", "content": answer})
        return answer

    def execute(self) -> str:
        completion = self.client.chat.completions.create(
                        model=self.model,
                        temperature=0,
                        messages=self.messages)
        return completion.choices[0].message.content


if __name__ == "__main__":
    agent = Agent()

    q = "Who are the top influencers of cyclotron funding?"
    print(f"> Question: {q}\n> Answer: {agent(q)}")

    q = "And in the context of 1930s, related to the Rockefeller Foundation?"
    print(f"> Question: {q}\n> Answer: {agent(q)}")