import streamlit as st

from langchain_ollama.llms  import OllamaLLM
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import (
    HumanMessage,
    AIMessage
)
from langchain_core.runnables.base import RunnableSequence
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

def set_page_ui():
    """
    Function where we define our main Ui components and page design
    """
    # Setting the page title
    st.set_page_config(
        page_title="Chess Master AI",
        page_icon="♟️", 
        initial_sidebar_state="auto", 
        layout="centered"
        )
    # Used to hide the image expand button
    st.markdown(
        """
    <style>
        [data-testid="StyledFullScreenButton"] {
            display: none
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Defining our side bar
    st.sidebar.image('./assets/img/chessMaster.webp', width=300)
    st.sidebar.title("Chess Tips & Guides")
    st.sidebar.markdown("""
    - **Basic Rules**: Learn the basics like piece movement and checkmate.
    - **Openings**: Explore common opening strategies.
    - **Endgames**: Master the endgame with expert guidance.
    """)

    # Title and description
    st.title("Chess Master AI ♟️")
    # User input section
    st.caption(" Ask me anything about chess !")

    # Footer
    st.sidebar.markdown(
        """
        <div style='text-align: center;'>
            <hr style='width:100%;'>
            <p style='color:gray; font-size:13px' >Built using Ollama, langchain and Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# Loading the model
@st.cache_resource
def load_llm():
    llm = OllamaLLM(
        model="gemma2"
    )
    return llm


store = {}
full_response = ""
set_page_ui()
llm = load_llm()
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

def init(
        llm : OllamaLLM
) -> RunnableSequence:
    """
    Initializes the main components required for generating responses to chess-related questions
    by creating a prompt template, loading embeddings, and setting up a retriever.
    """

    # Create the prompt template that will be used to structure user input and the model's response.
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

    # Create a prompt object from the template.
    prompt = ChatPromptTemplate.from_template(template)

    # Create embeddings using nomic-embed-text to convert text into a numerical format for similarity search.
    embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=False)

    # Load a persistent database of chess-related text (stored in ./ChessLLM/chessDB) with the provided embedding function.
    db = Chroma(persist_directory="./ChessLLM/chessDB",
                embedding_function=embeddings)

    # Create retriever
    retriever = db.as_retriever(
        search_type="similarity",  # Use similarity-based search for retrieving relevant information.
        search_kwargs= {"k": 3}    # Retrieve the top 5 most similar results.
    )

    chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    )

    return chain
    
def generate_responses(chain, completion, status):
    global full_response
    status.write(" Generating best possible response ...")
    
    first_item = None  # Variable to hold the first item
    for i, text in enumerate(chain.stream(completion)):
        response = text or ""
        full_response += response

        # Capture the first item and yield it twice
        if i == 0:
            first_item = response
            yield first_item
            yield first_item
        
        # Continue yielding the remaining items as normal
        else:
            yield response

def initialize_messages():
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Display chat history
    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("AI"):
                st.markdown(msg.content)

def main():
    chain = init(llm)
    initialize_messages()
    # User input
    if input_text := st.chat_input("You:"):
        st.session_state.messages.append(HumanMessage(content=input_text))
        # Display user input immediately
        with st.chat_message("human"):
            st.markdown(input_text)

        with st.chat_message("ai"):
                with st.status(" Wait for it...", expanded=False) as status:
                    status.write(" Retrieving relevant information from DBs...")
                    stream = generate_responses(chain, input_text, status)
                    # Cleanup the response
                    first_response = next(stream)  # Get the first response
                    status.write(" Writing...")
                # Now continue streaming the remaining responses
                st.write_stream(stream=stream)
                # Add AI response to history only if it is not empty
                if full_response.strip():  # Ensure the response is not just whitespace
                    st.session_state.messages.append(AIMessage(content=full_response))


if __name__ == "__main__":
    main()
