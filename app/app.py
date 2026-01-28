import streamlit as st
from pipeline.pipeline import AnimeRecommendationPipeline
from dotenv import load_dotenv

st.set_page_config(page_title="RAG Anime Recommender", layout="wide")

load_dotenv()

@st.cache_resource
def init_pipeline():
    return AnimeRecommendationPipeline()

pipeline=init_pipeline()

st.title("AI Anime Recommendation")
st.write("By: Atharv")

query=st.text_input("Enter Your Anime Preferences: eg Adventure anime with action")

if query:
    with st.spinner("Fetching Recommendations for you..."):
        response=pipeline.recommend(query)
        st.markdown("### Recommendations")
        st.write(response)