# ✅ vector.py – Create FAISS index from Top_10000_Movies_IMDb.csv

import os
import pandas as pd
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document

# ✅ Load API key from .env or Streamlit secrets
try:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("No API key in .env")
except:
    import streamlit as st
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# ✅ Load CSV
df = pd.read_csv("Top_10000_Movies_IMDb.csv")

# ✅ Build document list from correct columns
documents = []
for _, row in df.iterrows():
    metadata = row.to_dict()
    content = f"""
Title: {row['Movie Name']}
Genre: {row['Genre']}
Rating: {row['Rating']}
Plot: {row['Plot']}
Director(s): {row['Directors']}
Stars: {row['Stars']}
"""
    documents.append(Document(page_content=content, metadata=metadata))

# ✅ Split text into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = splitter.split_documents(documents)

# ✅ Generate embeddings and save FAISS index
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(splits, embeddings)
vectorstore.save_local("faiss_index")

print("✅ FAISS vector store saved successfully to 'faiss_index/'")
