# modules/tools.py

import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults

data_pdf_path = os.path.join(os.path.dirname(__file__), "..", "data", "data.pdf")

# Load and process the PDF
loader = PyPDFLoader(data_pdf_path)
docs = loader.load()

# Split the text into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
)
splits = text_splitter.split_documents(docs)

# Create an in-memory vector store from the document chunks
vectorstore = InMemoryVectorStore.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings()
)

# Create the retriever from the vector store
retriever = vectorstore.as_retriever()

# Define the PDF retriever tool
pdf_retriever_tool = create_retriever_tool(
    retriever,
    "pdf_retriever_tool",
    "Searches and returns context from the given PDF."
)

# Define the web search tool
search_tool = TavilySearchResults(max_results=3)

# Export the tools as a list for easy import
tools = [
    pdf_retriever_tool,
    search_tool,
]
