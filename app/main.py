import os
import streamlit as st
import time
from langchain.llms import Cohere
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI


from dotenv import load_dotenv, find_dotenv

load_dotenv()  # take environment variables from .env

st.title("NewBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")


main_placeholder = st.empty()
# llm = Cohere()
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)


if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","], chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    # create embeddings and save it to a folder
    embeddings = CohereEmbeddings()

    persist_dir = "db"
    vectordb = Chroma.from_documents(
        documents=docs, embedding=embeddings, persist_directory=persist_dir
    )
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    vectordb.persist()
    # vectordb = None


query = main_placeholder.text_input("Question: ")
if query:
    persist_dir = "db"
    vectordb = Chroma(
        persist_directory=persist_dir, embedding_function=CohereEmbeddings()
    )

    retriever = vectordb.as_retriever()
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)

    result = chain({"question": query}, return_only_outputs=True)
    # result will be a dictionary of this format --> {"answer": "", "sources": [] }
    st.header("Answer")
    st.write(result["answer"])

    # Display sources, if available
    sources = result.get("sources", "")
    if sources:
        st.subheader("Sources:")
        sources_list = sources.split("\n")  # Split the sources by newline
        for source in sources_list:
            st.write(source)
