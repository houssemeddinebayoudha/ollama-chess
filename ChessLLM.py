import streamlit as st

from typing import Generator
from langchain_ollama.llms  import OllamaLLM
from langchain_core.language_models import BaseLLM
from langchain_core.runnables import RunnablePassthrough
from streamlit.delta_generator import DeltaGenerator
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import (
    HumanMessage,
    AIMessage
)
from langchain_core.runnables.base import RunnableSequence
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma

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
@st.cache_resource # Decorator to cache functions that return global resources
def load_llm(
) -> BaseLLM:
    """
    Loads the gemma2 model localy using Ollama and the langchain's library

    Returns:
        llm (BaseLLM) : A BaseLLM that represent the large language model loaded
    """
    llm = OllamaLLM(
        model="gemma2"
    )
    return llm


store = {}
full_response = ""
set_page_ui()
llm = load_llm()

def init(
        llm : BaseLLM
) -> RunnableSequence:
    """
    Initialize Components 

    Args:
        llm (BaseLLM) : A BaseLLM that represent the large language model loaded using Langchain's library

    Returns:
        Chain : Represents the Langchain chain

    Functionality:
    - Initializes the main components required for generating responses to chess-related questions
    by creating a prompt template, loading embeddings, and setting up a retriever.
    """

    # Create the prompt template that will be used to structure user input and the model's response.
    template = """<bos><start_of_turn>user
    You are an expert in chess. Answer the following question directly and confidently based on the provided context. 
    Do not refer to the context or use phrases like "the text recommends."
    Simply provide your answer clearly and concisely.
    For example, if asked about a move, respond with: "To execute a ...." 
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
    
def generate_responses(
        chain : RunnableSequence,
        completion : str,
        status : DeltaGenerator) -> Generator[str, str, str]:
    """
    Generates responses from the given chain and yields them one by one.

    Args:
        chain (RunnableSequence): A RunnableSequence responsible for streaming completion results.
        completion (str): The completion request or input that is being processed.
        status (Streamlit Status): A status object used to update the UI (e.g., displaying progress).
    
    Yields:
        str: The generated response text
    
    Functionality:
    - The function streams completion results and yields them one by one.
    - The first response item is captured and yielded twice.
    - For each subsequent item, the response is yielded once as normal.
    """
    global full_response # Using a global variable to store the full accumulated response
    status.write(" Generating best possible response ...")
    
    first_item = None  # Variable to hold the first item
    for i, text in enumerate(chain.stream(completion)):
        response = text or ""
        full_response += response

        # Capture the first item and yield it twice
        if i == 0:
            first_item = response
            yield first_item
            yield first_item # Yield the first item again for special handling
        
        # Continue yielding the remaining items as normal
        else:
            yield response

def initialize_messages():
    """
    Initializes the chat history stored in the Streamlit session state if it doesn't exist.
    Then, iterates through the chat messages and displays them based on whether they are
    from a human or an AI.
    """
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
    # Initialization
    chain = init(llm)
    initialize_messages()
    # User input
    if input_text := st.chat_input("You:"):
        st.session_state.messages.append(HumanMessage(content=input_text))
        # Display user input immediately
        with st.chat_message("human"), st.empty():
            st.markdown(input_text)

        with st.chat_message("ai"), st.empty():
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
        st.empty()


if __name__ == "__main__":
    main()
