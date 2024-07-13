import os
import streamlit as st
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

# Initialize session state variables
if "response" not in st.session_state:
    st.session_state.response = None
if "source_documents" not in st.session_state:
    st.session_state.source_documents = None
if "memory" not in st.session_state:
    st.session_state.memory = None

# Streamlit UI
st.title("PDF Chatbot")

# Input fields for API key, PDF upload, and query
api_key = st.text_input("Enter your OpenAI API key", type="password")
pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
query = st.text_input("Enter your query")

# Button to clear previous response
if st.button("Clear Response"):
    st.session_state.response = None
    st.session_state.source_documents = None
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"  # Explicitly set the output key to 'answer'
    )

if api_key and pdf_file and query:
    os.environ['OPENAI_API_KEY'] = api_key
    
    # Save uploaded file temporarily
    with open("uploaded_pdf.pdf", "wb") as f:
        f.write(pdf_file.getbuffer())

    # Load PDF file
    loader = PyPDFLoader("uploaded_pdf.pdf")
    documents = loader.load()

    # Create text splitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=150
    )

    # Split document into chunks with splitter
    docs = text_splitter.split_documents(documents)

    # Create embeddings (Vector Representations) for chunks
    embeddings = OpenAIEmbeddings()

    # Store embeddings in Vector Database
    vectorDb = Chroma.from_documents(docs, embeddings)

    # Use existing memory or create a new one if not available
    if st.session_state.memory is None:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"  # Explicitly set the output key to 'answer'
        )

    # Create llm instance
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Create Conversational Retrieval Chain
    chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectorDb.as_retriever(search_type="mmr"),
        memory=st.session_state.memory,
        return_source_documents=True,
        output_key='answer'  # Explicitly set the output key to 'answer'
    )

    # Pass Human Question to Chain
    response = chain({"question": query})

    # Store response in session state
    st.session_state.response = response['answer']
    st.session_state.source_documents = response['source_documents']

    # Display response
    st.write("Response:", st.session_state.response)


