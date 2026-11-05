from langchain.agents import create_agent
from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
import os

load_dotenv()


llm = ChatOpenRouter(
    model="gpt-4o-mini",
    api_key=os.environ.get("OPENROUTER_API_KEY"))


agent = create_agent(model=llm,
                     system_prompt="You are a helpful assistant that helps details about Casino games",
                    checkpointer=InMemorySaver())


while True:
    user_input = input("user: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = agent.invoke({"messages":{"role":"user", "content":user_input}},
                            config={"configurable": {"thread_id": 1}})
    print("Agent:", response["messages"][-1].content)

