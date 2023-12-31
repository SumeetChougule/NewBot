import pickle
import os
import time
import langchain
from langchain.llms import Cohere
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import CohereEmbeddings

# from langchain.vectorstores.faiss import FAISS
# from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from dotenv import find_dotenv, load_dotenv

# Load environment variables
load_dotenv()


# Initialize LLM with required params
llm = Cohere()
llm("what is gravity?")

loader = UnstructuredURLLoader(
    urls=[
        "https://www.moneycontrol.com/news/business/banks/hdfc-bank-re-appoints-sanmoy-chakrabarti-as-chief-risk-officer-11259771.html",
        "https://www.moneycontrol.com/news/business/markets/market-corrects-post-rbi-ups-inflation-forecast-icrr-bet-on-these-top-10-rate-sensitive-stocks-ideas-11142611.html",
    ]
)
data = loader.load()
len(data)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

docs = text_splitter.split_documents(data)
len(docs)
docs[9]


embeddings = CohereEmbeddings()

# vectorindex_cohere = FAISS.from_documents(docs, embeddings)

# vectorindex_openai = FAISS.from_documents(docs, emb)

# instructor_embeddings = HuggingFaceInstructEmbeddings()

persist_dir = "db"

vectordb = Chroma.from_documents(
    documents=docs, embedding=embeddings, persist_directory=persist_dir
)

vectordb.persist()
vectordb = None

vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

retriever = vectordb.as_retriever()

chain = RetrievalQAWithSourcesChain.from_llm(
    llm=llm, retriever=retriever, return_source_documents=True
)
chain


query = "what is the price of Tiago iCNG?"
query = "WHO IS THE CRO?"
# query = "what are the main features of punch iCNG?"

langchain.debug = True

chain({"question": query}, return_only_outputs=True)
chain("what is the article about?")
