# ✅ visualize.py – Fast, Enhanced Movie Vector Explorer

import os
import pandas as pd
import umap
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# ✅ Load OpenAI API Key
try:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("No API key in .env")
except:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# ✅ Streamlit setup
st.set_page_config(page_title="📊 Movie Vectors", layout="wide")
st.title("📊 IMDb Movie Vector Explorer (Fast Mode)")
st.markdown("Explore top movie vectors with UMAP + Plotly. Filters enabled!")

# ✅ Load and limit data
@st.cache_resource
def load_data(sample_size=1000):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    df = pd.read_csv("Top_10000_Movies_IMDb.csv").head(sample_size).copy()
    vectors = vectorstore.index.reconstruct_n(0, len(df))
    return df, vectors

df, vectors = load_data(sample_size=1000)

# ✅ UMAP
@st.cache_resource
def reduce(vectors):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine")
    coords = reducer.fit_transform(vectors)
    return coords

coords = reduce(vectors)
df["x"] = coords[:, 0]
df["y"] = coords[:, 1]

# ✅ Filter UI
genre_filter = st.multiselect("🎭 Genre", sorted(df["Genre"].unique()))
rating_range = st.slider("⭐ IMDb Rating", 0.0, 10.0, (7.0, 10.0), step=0.1)
search = st.text_input("🔍 Search by title or plot")

# ✅ Filter Logic
filtered = df[
    df["Rating"].between(rating_range[0], rating_range[1])
]
if genre_filter:
    filtered = filtered[filtered["Genre"].isin(genre_filter)]
if search:
    search_lower = search.lower()
    filtered = filtered[
        filtered["Movie Name"].str.lower().str.contains(search_lower) |
        filtered["Plot"].str.lower().str.contains(search_lower)
    ]

# ✅ Add color bucket
def rating_bucket(r):
    if r >= 8.5: return "🔥 Masterpiece"
    elif r >= 8.0: return "🌟 Great"
    elif r >= 7.5: return "👍 Good"
    else: return "🙂 Watchable"

filtered["Rating Level"] = filtered["Rating"].apply(rating_bucket)

# ✅ Plotly chart
fig = px.scatter(
    filtered,
    x="x", y="y",
    color="Rating Level",
    hover_data=["Movie Name", "Rating", "Genre", "Plot"],
    title=f"🎬 Movie Embeddings (Top {len(df)} sample)",
    opacity=0.75,
    height=700
)
st.plotly_chart(fig, use_container_width=True)
