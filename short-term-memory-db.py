from langchain.agents import create_agent
from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver
import os

load_dotenv()


llm = ChatOpenRouter(
    model="gpt-4o-mini",
    api_key=os.environ.get("OPENROUTER_API_KEY"))

DB_URL = os.environ.get("DATABASE_URL")

with PostgresSaver.from_conn_string(DB_URL) as checkpointer:
    checkpointer.setup()
    agent = create_agent(model=llm,
                        system_prompt="You are a helpful assistant that helps details about Casino games",
                        checkpointer=checkpointer)


    while True:
        user_input = input("user: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = agent.invoke({"messages":{"role":"user", "content":user_input}},
                                config={"configurable": {"thread_id": 1}})
        print("Agent:", response["messages"][-1].content)

