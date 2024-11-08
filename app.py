import os

from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"] = openai_api_key

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

from typing import Sequence

import bs4

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from typing_extensions import Annotated, TypedDict

# Imports features
import modules.agents as agents

def main():
    # Define test profile with default values
    test_profile = {
        "messages": [],
        "language": "English",  # Default language
        "age": 25,              # Default age
        "name": "Bob",          # Default name
        "preferences": "Cowboy Speak" # Default preferences
    }

    print("Enter text (type 'exit' to quit):")
    
    while True:
        user_input = input(">> ")
        
        if user_input.lower() == 'exit':  # Exit condition
            print("Exiting program.")
            break
        else:
            # Append the user input to the messages list
            test_profile["messages"].append(user_input)
            
            # Create the state object
            state = {
                "messages": test_profile["messages"],
                "language": test_profile["language"],
                "age": test_profile["age"],
                "name": test_profile["name"],
                "preferences": test_profile["preferences"],  # Use style instead of preferences
            }
            
            # Call the model with the state
            response = agents.call_model(state)

            print("AI Response:", response)

if __name__ == "__main__":
    main()