from langchain.tools import tool
from langchain.chat_models import init_chat_model


# ----------------------------
# Define Tools
# ----------------------------
@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"The weather in {location} is sunny."


@tool
def calculator(expression: str) -> str:
    """Perform mathematical calculations."""
    return str(eval(expression))


# ----------------------------
# Initialize LLM
# ----------------------------
llm = init_chat_model(
    model="ollama:qwen3-vl:8b"
)

# ----------------------------
# Bind Tools
# ----------------------------
tools = [get_weather, calculator]

llm_with_tools = llm.bind_tools(tools)

# ----------------------------
# Ask Question
# ----------------------------
query = "What is the weather in Patna?"

response = llm_with_tools.invoke(query)

print("LLM Response:")
print('-'*30)
print(response)
print("\nTool Calls:")
print('-'*30)
print(response.tool_calls)

# ----------------------------
# Execute Tool (manually)
# ----------------------------
tool_map = {
    "get_weather": get_weather,
    "calculator": calculator
}

for tool_call in response.tool_calls:

    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    result = tool_map[tool_name].invoke(tool_args)

    print("\nTool Executed:")
    print('-' * 30)
    print(f"Tool: {tool_name}")
    print('-' * 30)
    print(f"Result: {result}")