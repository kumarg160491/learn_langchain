from langchain.agents import create_agent
from langchain.tools import tool
from agent5_langgraph.checkpoint.memory import InMemorySaver
from langchain_core.utils.uuid import uuid7

@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"The weather is sunny in {city}"

agent = create_agent(
    model="ollama:qwen3-vl:8b",
    tools=[get_weather],
    checkpointer=InMemorySaver(),
    system_prompt="You are a helpful assistant",
)

config = {
    "configurable": {
        "thread_id": str(uuid7())
    }
}

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What's the weather in San Francisco?"
            }
        ]
    },
    config=config,
)

print(result["messages"][-1].content)
print("-"*50)
print(result)