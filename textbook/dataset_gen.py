import openai


class OpenAIGenerator():

    def __init__(self, model: str = "gpt-3.5-turbo"):

        self.model = model
    
    def generate(self, prompt: str) -> str:
        chat_completion = openai.ChatCompletion.create(model=self.model, messages=[{"role": "user", "content": "Hello world"}])
        return chat_completion.choices[0].message.contentss