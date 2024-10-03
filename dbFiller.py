from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS



# Load documents from a directory
loader = PyPDFDirectoryLoader("./pdfs")


# Loading the data into a document object
documents = loader.load()

print("********** Data Loaded loaded **********")
print(f"Loaded {len(documents)} documents.")  # Check the number of loaded documents


# Loading the nomi-embed-text open embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)

# Creating a recursive text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300,
    add_start_index=True,
)

# Split documents into chunks
texts = text_splitter.split_documents(documents)
# Create vector store
vectorstore = FAISS.from_documents(
    embedding=embeddings,
    documents=texts,
    )

vectorstore.save_local("chess_db")
print("Vectorstore created")