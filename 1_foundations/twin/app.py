import os

os.environ.pop("SSLKEYLOGFILE", None)

from openai import OpenAI
from context import TWIN_SYSTEM_PROMPT
from tools import tools, handle_tool_calls
from styles import CSS, JS, EXAMPLES
from dotenv import load_dotenv
import gradio as gr

load_dotenv(override=True)

OLLAMA_BASE_URL = "https://ollama.com/v1"

ollama_api_key = os.getenv('OLLAMA_API_KEY')

if ollama_api_key:
    print(f"Ollama API Key exists and begins {ollama_api_key[:6]}")
else:
    print("Ollama API Key not set (and this is optional)")

ollama = OpenAI(base_url=OLLAMA_BASE_URL, api_key=ollama_api_key)

#MODEL_NAME = "gpt-5.4-mini"

MODEL_NAME = "gemma4:31b"

#openai = OpenAI()

system = [{"role": "system", "content": TWIN_SYSTEM_PROMPT}]


def chat(message, history):
    messages = system + history + [{"role": "user", "content": message}]
    #response = openai.chat.completions.create(model=MODEL_NAME, messages=messages, tools=tools)
    response = ollama.chat.completions.create(model=MODEL_NAME, messages=messages, tools=tools)
    while response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        tool_calls = message.tool_calls
        results = handle_tool_calls(tool_calls)
        messages.append(message)
        messages.extend(results)
        #response = openai.chat.completions.create(model=MODEL_NAME, messages=messages, tools=tools)
        response = ollama.chat.completions.create(model=MODEL_NAME, messages=messages, tools=tools)
    return response.choices[0].message.content


if __name__ == "__main__":
    gr.ChatInterface(
        chat,
        examples=EXAMPLES,
        title="Digital Twin",
        description="Talk to my AI twin about my career",
        chatbot=gr.Chatbot(show_label=False),
    ).launch(css=CSS, js=JS, theme=gr.themes.Base())
