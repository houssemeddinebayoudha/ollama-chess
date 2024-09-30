from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama

from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser


# # Create embeddingsclear
embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=False)

db = Chroma(persist_directory="./ChessLLM/chessDB",
            embedding_function=embeddings)

# # Create retriever
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs= {"k": 5}
)

# # Create Ollama language model - Gemma 2
local_llm = 'gemma2'

llm = ChatOllama(model=local_llm,
                 keep_alive="5m", 
                 max_tokens=512,  
                 temperature=0)

# Create prompt template
template = """<bos><start_of_turn>user
You are an expert in chess. Answer the following question directly and confidently based on the provided context. 
Do not refer to the context or use phrases like "the text recommends." Simply provide your answer clearly and concisely.
For example, if asked about a move, respond with: "To execute a castling move, the rook moves to the square next to the king." 
Use full sentences with correct spelling and punctuation. If applicable, use lists to present information.

CONTEXT: {context}

QUESTION: {question}

<end_of_turn>
<start_of_turn>model
ANSWER:"""



prompt = ChatPromptTemplate.from_template(template)

# Create the RAG chain using LCEL with prompt printing and streaming output
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# Function to ask questions
def ask_question(question):
    print("Answer:\n\n", end=" ", flush=True)
    for chunk in rag_chain.stream(question):
        print(chunk.content, end="", flush=True)
    print("\n")

# Example usage
if __name__ == "__main__":
    while True:
        user_question = input("Ask a question (or type 'quit' to exit): ")
        if user_question.lower() == 'quit':
            break
        answer = ask_question(user_question)
        # print("\nFull answer received.\n")