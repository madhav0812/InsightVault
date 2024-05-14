import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import os

load_dotenv()
api_key=os.environ["OPENAI_API_KEY"]



def extract_text_from_pdfs(pdf_docs):
    text = ''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def split_text_into_chunks(raw_text):
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(raw_text)
    return chunks

def create_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(api_key=api_key)  # Pass the API key here
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def initialize_conversation_chain(vectorstore):
    # Ensure that the API key is passed to the ChatOpenAI class
    llm = ChatOpenAI(api_key=api_key)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
    return conversation_chain



def handle_user_input(user_question):
    response = st.session_state.lagozon_conversation({'question': user_question})
    st.write(response['answer'])


def main():
    st.set_page_config(page_title='InsightVault', page_icon=":books:", layout='wide')

    if "lagozon_conversation" not in st.session_state:
        st.session_state.lagozon_conversation = None

    st.header("InsightVault :books:")
    user_question = st.text_input("Ask a question about your data: ")

    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.subheader("Upload PDF files")
        pdf_docs = st.file_uploader("Upload files here and click process", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = extract_text_from_pdfs(pdf_docs)
                text_chunks = split_text_into_chunks(raw_text)
                vectorstore = create_vectorstore(text_chunks)
                st.session_state.lagozon_conversation = initialize_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()
