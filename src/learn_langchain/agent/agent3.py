from langchain_core.messages import *
from langchain.agents import *
from langchain.chat_models import init_chat_model

agent = init_chat_model(
    model="ollama:qwen3-vl:8b",
)

messages = [
    SystemMessage('You are a helpful assistant'),
    HumanMessage('Can you help me?'),
    AIMessage('Sure I can help you in solving the issue'),
    HumanMessage('Great! what is 2+2?')
]

response = agent.invoke(messages)
response = messages.append(response)
print(response)