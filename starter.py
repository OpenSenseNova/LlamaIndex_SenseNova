import logging
import sys
import os

# Import necessary modules and classes
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter

from sensecore import SenseNova
from novaembedding import SenseNovaEmbedding

# Configure logging to output to standard output and set log level to DEBUG
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Initialize the SenseNova model
# llm = OpenAI(temperature=0.1, model="gpt-4o")
llm = SenseNova(model="SenseChat-32K", temperature=0.1)

# Set global LLM and embedding model
Settings.llm = llm
Settings.embed_model = SenseNovaEmbedding()

# Define the storage directory
PERSIST_DIR = "./storage"

# Check if the storage directory exists
if not os.path.exists(PERSIST_DIR):
    # If the storage directory does not exist, load documents and create an index
    documents = SimpleDirectoryReader("data").load_data()
    # Use SentenceSplitter to split documents into nodes
    splitter = SentenceSplitter(chunk_size=80, chunk_overlap=20, include_metadata=False)
    print(splitter)
    nodes = splitter.get_nodes_from_documents(documents)

    # Print the text of each node
    for i, node in enumerate(nodes):
        print(f'[Node {i}]：{node.text}')
    
    # Create a vector store index
    index = VectorStoreIndex(nodes=nodes, insert_batch_size=1)
    # Store the index for later use
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # If the storage directory exists, load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# Create a query engine and perform a query
query_engine = index.as_query_engine()
response = query_engine.query(
    "What did the author do at IBM?"
)
print(response)
