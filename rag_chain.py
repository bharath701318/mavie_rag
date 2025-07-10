# ✅ rag_chain.py – Standalone RAG logic with FAISS and LangChain
import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# ✅ Load OpenAI API Key from .env or fallback to Streamlit secrets
try:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("No API key in .env")
except:
    import streamlit as st
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# ✅ Load FAISS vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

# ✅ Set up RAG chain
llm = ChatOpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# ✅ Example usage
if __name__ == "__main__":
    while True:
        query = input("\n🎤 Ask something about a movie (or type 'exit'): ")
        if query.lower() == "exit":
            break
        response = qa_chain.run(query)
        print(f"💬 Answer: {response}")
