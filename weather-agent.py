from langchain.agents import create_agent
from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
import os
import requests
from langchain_tavily import TavilySearch

load_dotenv()


llm = ChatOpenRouter(
    model="gpt-4o-mini",
    api_key=os.environ.get("OPENROUTER_API_KEY"))

@tool
def get_weather(city: str) -> str:

    """
    Weather Agent to get the real time current weather information
    """

    url="https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": city,
        "appid": os.getenv("WEATHER_API_KEY"),
        "units": "metric"
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        return f"Unable to fetch data from weather api"
    return response.json()


tavily = TavilySearch(max_results= 4)




agent = create_agent(model=llm,
                     system_prompt="You are a helpful assistant that helps details about " \
                     "current weather or regular live search based on the user input",
                    checkpointer=InMemorySaver(),
                    tools=[get_weather, tavily])


while True:
    user_input = input("user: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = agent.invoke({"messages":{"role":"user", "content":user_input}},
                            config={"configurable": {"thread_id": 1}})
    print("Agent:", response["messages"][-1].content)

