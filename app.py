import streamlit as st
import os
import time
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharecterTextSplitter
from langchain.chains.combine_documents import create_stuffs_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorestores import FAISS

from dotenv import load_dotenv
load_dotenv()


os.environ['NVIDIA_API_KEY']=os.getenv('NVIDIA_API_KEY')

llm=ChatNVIDIA(model="meta/llama3-8b-instruct")

#function is created for performing embedding on the pdfs present under RAG_DOCS folder
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=NVIDIAEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader("./files")
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=700,chunk_overlap=50)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

st.title("NVIDIA NIM demo")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
please provide the most accurate response base on the question
<context>
{context}
Questions:{input}
"""
)

prompt1=st.text_input("Enter your question from documents")

if st.button("Document embedding"):
    vector_embedding()
    st.write("FAISS vector store db is ready using nvidiaembedming")


if prompt1:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    response=retrieval_chain.invoke({'input':prompt1})
    print("Rsponse time:",time.process_time()-start)
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        for i,doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("---------------------")
