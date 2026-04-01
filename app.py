import streamlit as st
from recommender import retrieve, fetch_recommendations, generate_recommendation

st.set_page_config(page_title="CineMatch AI", page_icon="🎬", layout="centered")

st.title("🎬 CineMatch AI")
st.markdown("*Your personal AI-powered movie recommender*")
st.divider()

mood = st.text_input(
    "What are you in the mood for tonight?",
    placeholder="e.g. something light and funny to watch with family"
)

Viewers = st.selectbox(
    "Who's watching?",
    ("Just me", "With partner", "With family", "With friends")
)

Platform = st.selectbox(
    "Which platform do you prefer?",
    ("Any", "Netflix", "Prime", "Disney+")
)

if st.button("Find my match 🎯", type="primary"):
    if not mood:
        st.error("Please describe what you're in the mood for!")
    else:
        try:
            with st.spinner("CineMatch AI is finding your perfect watch..."):
                results = retrieve(mood)
                movies = fetch_recommendations(results)
                recommendation = generate_recommendation(mood, results, movies)
            
            st.divider()
            st.subheader("Your personalised recommendation")
            st.markdown(recommendation)
            
            st.divider()
            st.subheader("Based on your watch history")
            for meta in results["metadatas"][0]:
                st.markdown(f"- **{meta['title']}** — {meta['mood']} (your rating: {meta['rating']}/10)")
        
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.info("Please try again or rephrase your mood description.")