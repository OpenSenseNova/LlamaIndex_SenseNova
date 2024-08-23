import logging
import sys
import os

from llama_index.core import KeywordTableIndex, SimpleDirectoryReader,StorageContext,ServiceContext,load_index_from_storage
from llama_index.core import Settings
from sensecore import SenseNova

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))



# llm = OpenAI(temperature=0.1, model="gpt-4o")
llm = SenseNova(model="SenseChat-32K",temperature=0.1)

Settings.llm = llm

# check if storage already exists
PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents = SimpleDirectoryReader("data").load_data()
    index = KeywordTableIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# get response from query
query_engine = index.as_query_engine()
response = query_engine.query(
    "What did the author do at IBM?"
)
print(response)