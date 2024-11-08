import os

from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"] = openai_api_key

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

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
from typing import Sequence
from .tools import retriever
# from . import prompts
    
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer in {language}."
         "Treat user as age {age}."
         "They are named {name}."
         "They have the following preferences: {preferences}."
         ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

runnable = prompt | llm
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str
    age: int
    name: str
    preferences: str


workflow = StateGraph(state_schema=State)


def call_model(state: State):
    response = runnable.invoke(state)
    # Update message history with response:
    return {"messages": [response]}


workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc345"}}

# Define the callable function that can be used externally
def generate_response(messages: Sequence[str], language: str, age: int, name: str, style: str) -> str:
    """
    Generates a response from the AI model based on the provided inputs.
    
    Args:
        messages (Sequence[str]): The input messages to send to the AI.
        language (str): The language in which the AI should respond.
        age (int): The age of the user.
        name (str): The name of the user.
        style (str): The preferred style (e.g., formal, casual).

    Returns:
        str: The AI's response.
    """

    input_dict = {
        "messages": [HumanMessage(msg) for msg in messages],
        "language": language,
        "age": age,
        "name": name,
        "style": style,
    }

    # Invoke the app with the provided inputs
    output = app.invoke(input_dict, config)
    
    # Extract and return the last response message
    return output["messages"][-1]