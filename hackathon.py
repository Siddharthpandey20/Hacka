content=int(input("Choose between url(1) or pdf(2)"))
# Import required packages
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Configure Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

# Get API key from environment variable
gemini_api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=gemini_api_key)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=gemini_api_key)

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load a free embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)

import os
from getpass import getpass

os.environ["LANGSMITH_TRACING"] = "true"
langsmith_api_key = os.getenv('LANGSMITH_API_KEY')
os.environ["Langsmith"] = langsmith_api_key


"""#For Web  Url

first for updating
"""

import threading
vector_store_lock = threading.Lock()

def update_vector_store(file_paths=None, url=None):
    """Safely updates the vector database by replacing modified content instead of just appending."""
    with vector_store_lock:
        docs = []
        new_data = {}

        if file_paths:
            for path in file_paths:
                new_data[path] = load_local_files([path])

        if url:
            new_data[url] = load_web_content_with_links(url)

        # **Get existing document IDs from FAISS**
        existing_ids = set(vector_store.index_to_docstore_id.values())

        # **Remove old content before adding new**
        for doc_id, new_docs in new_data.items():
            if isinstance(new_docs, list) and len(new_docs) > 0:
                try:
                    if doc_id in existing_ids:
                        vector_store.delete([doc_id])
                except ValueError as e:
                    print(f"Warning: Could not delete {doc_id} - {e}")

                docs.extend(new_docs)

        if docs:
            try:
                vector_store.add_documents(docs)
                vector_store.save_local("vector_db")
                print("Vector database updated successfully!")
            except Exception as e:
                print(f"Error updating vector store: {e}")

import bs4, requests, time
from bs4 import BeautifulSoup
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing_extensions import List, TypedDict
from threading import Thread

def filter_main_content(html):
    """Extracts only relevant textual content from a webpage."""
    soup = BeautifulSoup(html, 'html.parser')

    # Remove scripts, styles, and footers
    for tag in soup(["script", "style", "footer", "nav", "header", "aside"]):
        tag.decompose()

    # Extract meaningful text
    text = ' '.join([p.get_text() for p in soup.find_all('p')])
    return text if text else soup.get_text()

def extract_hyperlinks(url, allowed_domains=None):
    """Extracts hyperlinks but filters out irrelevant links based on domain."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        links = [a['href'] for a in soup.find_all('a', href=True)]

        # Allow only relevant domains if specified
        if allowed_domains:
            valid_links = [link for link in links if any(domain in link for domain in allowed_domains)]
        else:
            valid_links = [link for link in links if link.startswith("http")]

        return valid_links
    except Exception as e:
        print(f"Error fetching links from {url}: {e}")
        return []

def load_web_content_with_links(url):
    """Loads main content from URL and relevant hyperlinks."""
    try:
        # Add verify=False for SSL certificate issues
        response = requests.get(url, verify=False)
        response.raise_for_status()
        filtered_content = filter_main_content(response.text)

        extracted_links = extract_hyperlinks(url, allowed_domains=[url.split('/')[2]])  # Restrict links to same domain
        web_paths = (url,) + tuple(extracted_links)

        loader = WebBaseLoader(web_paths=web_paths)
        docs = loader.load()

        if not docs:
            print("No content could be loaded from the URL.")
            return []

        # Replace raw HTML with extracted text
        for doc in docs:
            doc.page_content = filter_main_content(doc.page_content)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)

        # Start monitoring URL changes in the background
        Thread(target=monitor_urls, args=([url] + extracted_links,), daemon=True).start()

        return text_splitter.split_documents(docs)
    except Exception as e:
        print(f"Error loading content from {url}: {e}")
        return []

def monitor_urls(url_list, interval=600):
    """Fetches and updates web content every `interval` seconds."""
    while True:
        for url in url_list:
            update_vector_store(url=url)
        time.sleep(interval)  # Wait before next check
all_splits=[]
if content==1:
    url =input("Give the url")
    all_splits = load_web_content_with_links(url)

"""#For Local Document"""

from langchain_community.document_loaders import PyMuPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_local_files(file_paths):
    """Loads and processes local documents (PDF, CSV) for vector DB."""
    docs = []
    for file_path in file_paths:
        if file_path.endswith(".pdf"):
            loader = PyMuPDFLoader(file_path)
        elif file_path.endswith(".csv"):
            loader = CSVLoader(file_path)
        else:
            continue
        docs.extend(loader.load())

    # Split documents into smaller chunks for better vector retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    return text_splitter.split_documents(docs)

# Fix the document path (remove extra quotes and fix path separator)
split_docs=''
if content==2:
    doc_path = "D:\\Desktop\\Hackathon\\2. Candide, Voltaire.pdf"  # Use double backslashes for Windows paths
    split_docs = load_local_files([doc_path])

"""#**Updating** VECTOR DB when update in Document occur"""

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
import os

class FileChangeHandler(FileSystemEventHandler):
    """Monitors file changes and updates vector store automatically."""
    def on_modified(self, event):
        if event.src_path.endswith((".pdf", ".csv")):
            print(f"Detected change in {event.src_path}, updating vector DB...")
            update_vector_store(file_paths=[event.src_path])

# Get the directory containing the document instead of the file itself
if content==2:
    watchdog_path = os.path.dirname(doc_path)
    observer = Observer()
    event_handler = FileChangeHandler()
    observer.schedule(event_handler, path=watchdog_path, recursive=False)  # Set recursive to False for single directory
    observer.start()

# Index chunks
if not all_splits:
    vector_store = FAISS.load_local("vector_db", embeddings) if os.path.exists("vector_db") else FAISS.from_documents(split_docs, embeddings)

else:
    vector_store = FAISS.load_local("vector_db", embeddings) if os.path.exists("vector_db") else FAISS.from_documents(all_splits, embeddings)
"""#Using Hybrid Seaarch"""


from rank_bm25 import BM25Okapi
all_splits=split_docs+all_splits
# Store tokenized documents for BM25
tokenized_corpus = [doc.page_content.lower().split() for doc in all_splits]
bm25_index = BM25Okapi(tokenized_corpus)

def hybrid_search(query, top_n=10):
    """Filters using BM25, refines using FAISS vector search, and retrieves most relevant docs."""
    global bm25_index, tokenized_corpus, vector_store, all_splits

    # **Query Expansion (Fix AIMessage issue)**
    query_expansion_prompt = f"Expand this query '{query}' with specific details and Rank by specificity."
    expanded_query = llm.invoke(query_expansion_prompt)

    if hasattr(expanded_query, "content"):
        expanded_query = expanded_query.content
    elif isinstance(expanded_query, str):
        pass
    else:
        expanded_query = str(expanded_query)

    query_tokens = expanded_query.lower().split()

    # **Stage 1: BM25 Keyword Search**
    scores = bm25_index.get_scores(query_tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    candidate_docs = [all_splits[i] for i in top_indices]

    # **Stage 2: Vector Search (Retrieve More for Better Matching)**
    vector_results = vector_store.similarity_search_with_score(query, k=top_n * 2)  # Increased candidates

    # **Merge BM25 + FAISS results with Weighted Scoring**
    weighted_results = {}

    for doc, score in vector_results:
        weighted_results[doc.page_content] = (doc, score * 0.7)  # FAISS weighted at 70%

    for doc in candidate_docs:
        if doc.page_content in weighted_results:
            weighted_results[doc.page_content] = (doc, weighted_results[doc.page_content][1] + 0.3)  # BM25 weighted at 30%
        else:
            weighted_results[doc.page_content] = (doc, 0.3)

    # **Sort by Combined Score**
    def get_bm25_score(doc):
        bm25_scores = bm25_index.get_scores(doc.page_content.split())
        return float(bm25_scores.mean()) if bm25_scores.size > 0 else 0

    final_results = sorted(weighted_results.values(), key=lambda item: (item[1], get_bm25_score(item[0])), reverse=True)

    # **Return Final Retrieved Documents**
    return [doc[0] for doc in final_results[:top_n]]

from langgraph.graph import MessagesState, StateGraph

graph_builder = StateGraph(MessagesState)

from langchain_core.tools import tool
from typing import Any, Dict

@tool
def retrieve(query: str) -> str:  # Added type hint here
    """Retrieves most relevant information using hybrid search and re-ranks.

    Args:
        query: The search query to use for retrieval.

    Returns:
        Final ranked response as a string.
    """

    retrieved_docs = hybrid_search(query, top_n=10)  # Get more docs before re-ranking

    # **Re-rank using LLM to ensure best answer is selected**
    rerank_prompt = f"You are retrieving information for this query: '{query}'.\nPrioritize the most detailed and specific response  give higher importance to documents that create a logical, flowing narrative and prioritize documents that reflect the temporal progression of events rather than random moments:\n"
    for doc in retrieved_docs:
        rerank_prompt += f"- {doc.page_content[:200]}...\n"

    ranked_docs = llm.invoke(rerank_prompt)


    if hasattr(ranked_docs, "content"):
        ranked_docs = ranked_docs.content

    return ranked_docs  # Final ranked response

"""

Our graph will consist of three nodes:

1. A node that fields the user input, either generating a query for the retriever or responding directly;
2. A node for the retriever tool that executes the retrieval step;
3. A node that generates the final response using the retrieved context.
"""

from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode


# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}


# Step 2: Execute the retrieval.
tools = ToolNode([retrieve])


# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
    "You are an assistant for question-answering tasks. "
    "Strictly use only the retrieved context to answer the question. "
    "Do not use any external knowledge. "
    "include a short quote from the context to support your answer. "
    "Use a maximum of five sentences and keep the response clear and concise."
    "\n\n"
    f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}

from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# Specify an ID for the thread
config = {"configurable": {"thread_id": "abc123"}}

"""And when executing a search, we can stream the steps to observe the query generation, retrieval, and answer generation:"""

# input_message = "Where was Candide brought up, and why was he expelled?"
input_message=input("Enter Question")

for step in graph.stream(
    {"messages": [{"role": "user", "content": input_message}]},
    stream_mode="values",
    config=config
):
    step["messages"][-1].pretty_print()

# print(hybrid_search(query=input_message, top_n=5))
