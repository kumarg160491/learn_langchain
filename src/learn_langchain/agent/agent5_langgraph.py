from typing import Annotated

from langchain.tools import tool
from langchain.chat_models import init_chat_model

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import requests
from typing_extensions import TypedDict


# -------------------------
# Tools
# -------------------------


@tool
def get_weather(city: str) -> str:
    """Get the current weather of a city in India."""

    try:
        url = f"https://wttr.in/{city}?format=j1"

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()

        current = data["current_condition"][0]

        temperature = current["temp_C"]
        condition = current["weatherDesc"][0]["value"]
        humidity = current["humidity"]
        wind_speed = current["windspeedKmph"]

        return (
            f"Weather in {city}: "
            f"{condition}, "
            f"Temperature: {temperature}°C, "
            f"Humidity: {humidity}%, "
            f"Wind Speed: {wind_speed} km/h"
        )

    except Exception as e:
        return f"Unable to fetch weather data: {str(e)}"


@tool
def calculator(expression: str) -> str:
    """Perform mathematical calculations."""
    return str(eval(expression))


tools = [get_weather, calculator]


# -------------------------
# State
# -------------------------

class State(TypedDict):
    messages: Annotated[list, add_messages]


# -------------------------
# LLM
# -------------------------

llm = init_chat_model(
    model="ollama:qwen3-vl:8b"  # replace with your model
)

llm_with_tools = llm.bind_tools(tools)


# -------------------------
# Chatbot Node
# -------------------------

def chatbot(state: State):
    return {
        "messages": [
            llm_with_tools.invoke(state["messages"])
        ]
    }


# -------------------------
# Graph
# -------------------------

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)

graph_builder.add_node(
    "tools",
    ToolNode(tools)
)

graph_builder.add_edge(
    START,
    "chatbot"
)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition
)

graph_builder.add_edge(
    "tools",
    "chatbot"
)

graph_builder.add_edge(
    "chatbot",
    END
)

graph = graph_builder.compile()


# -------------------------
# Invoke
# -------------------------

response = graph.invoke(
    {
        "messages": [
            ("user", "What is the weather in Patna?")
        ]
    }
)

for message in response["messages"]:
    print(message)