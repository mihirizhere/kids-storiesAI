from dotenv import load_dotenv
import os
import openai

# Load variables from .env into os.environ
load_dotenv()

# Set API key once (no more exporting in your shell)
openai.api_key = os.getenv("OPENAI_API_KEY")

"""
Before submitting the assignment, describe here in a few sentences what you would have built next if you spent 2 more hours on this project:

"""

def call_model(prompt: str, max_tokens=500, temperature=0.1) -> str:
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message["content"]  # type: ignore

example_requests = "A story about a girl named Alice and her best friend Bob, who happens to be a cat."

def main():
    # confirm key loaded
    print("Key is:", os.getenv("OPENAI_API_KEY")[:6] + "…")
    user_input = input("What kind of story do you want to hear? ")
    response = call_model(user_input)
    print(response)

if __name__ == "__main__":
    main()
