from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_prompt = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

chat_history = []

with open('C:\\Users\\kumar\\OneDrive\\Documents\\Learning\\Langchain\\learn_langchain\\src\\learn_langchain\\chatmodels\\chat_history.txt') as f:
    chat_history.extend(f.readlines())

prompt = chat_prompt.invoke({'chat_history': chat_history, 'query': 'where is my refund'})
print(prompt)