# ✅ app.py – Streamlit UI for RAG Movie Chatbot

import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ✅ Load OpenAI API key (from .env or st.secrets)
try:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("No API key in .env")
except:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# ✅ Streamlit app config
st.set_page_config(page_title="🎬 MovieBot - IMDb Chat", page_icon="🎥")
st.title("🎬 MovieBot: Chat with Top 10K IMDb Movies")
st.markdown("Ask me anything about the top 10,000 IMDb movies!")

# ✅ Load FAISS vector store
@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()
    return retriever

retriever = load_vectorstore()
llm = ChatOpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# ✅ Chat input
query = st.text_input("💬 Ask a question about any movie:")
if query:
    with st.spinner("🔍 Searching..."):
        response = qa_chain.run(query)
    st.success(response)
