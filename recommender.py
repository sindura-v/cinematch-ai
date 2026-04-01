import chromadb
from openai import OpenAI
from dotenv import load_dotenv
import os
import requests

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="watchlist")

def retrieve(mood_query, n_results=3):
    # 1. Embed the mood query — same way you embedded movies
    response = client.embeddings.create(
        input=mood_query,
        model="text-embedding-3-small"
    )
    embedding = response.data[0].embedding

    # 2. Query ChromaDB for the 3 most similar entries
    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
        where={"rating": {"$gte": 6}}
    )
    
    return results


TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE = "https://api.themoviedb.org/3"

GENRE_MAP = {
    "Comedy": 35,
    "Drama": 18,
    "Action": 28,
    "Thriller": 53,
    "Crime": 80,
    "Mystery": 9648,
    "Romance": 10749,
    "Horror": 27,
    "Animation": 16
}

def fetch_recommendations(retrieved_results, n=5):
    # Extract genre from top match
    top_genre = retrieved_results["metadatas"][0][0]["genre"]
    
    # Find matching TMDB genre ID
    genre_id = None
    for key in GENRE_MAP:
        if key.lower() in top_genre.lower():
            genre_id = GENRE_MAP[key]
            break
    
    if not genre_id:
        genre_id = 35  # default to Comedy
    
    # Use discover endpoint
    url = f"{TMDB_BASE}/discover/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "with_genres": genre_id,
        "vote_average.gte": 7,
        "primary_release_date.gte": "2020-01-01",
        "sort_by": "popularity.desc",
        "language": "en-US"
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    movies = []
    for item in data.get("results", [])[:n]:
        date = item.get("release_date") or item.get("first_air_date", "")
        tmdb_rating = item.get("vote_average", 0)  

        if tmdb_rating > 6.5:  
            movies.append({
                "title": item.get("title") or item.get("name"),
                "overview": item.get("overview", "")[:200],
                "tmdb_rating": item.get("vote_average", 0),
                "year": date[:4] if date else "Unknown"
            })

        if len(movies) >= n:
            break
    
    return movies
def generate_recommendation(mood, retrieved_results, tmdb_movies):
    
    # Format watchlist context
    watchlist_context = ""
    for i, meta in enumerate(retrieved_results["metadatas"][0]):
        watchlist_context += f"- {meta['title']} (rated {meta['rating']}/10, mood: {meta['mood']})\n"
    
    # Format TMDB movies
    tmdb_context = ""
    for movie in tmdb_movies:
        tmdb_context += f"- {movie['title']} ({movie['year']}, TMDB rating: {movie['tmdb_rating']}): {movie['overview']}\n"
    
    # Build the prompt
    user_message = f"""
        The user wants: {mood}

        Based on their watch history, they enjoy:
        {watchlist_context}

        Available movies/shows to recommend from:
        {tmdb_context}

        Write a personalised recommendation.
    """
    system_prompt = f"""
        You are CineMatch AI, an expert, enthusiastic film curator. 
        Your goal is to provide personalized, high-quality movie recommendations based only on the provided context 
        and the user's explicit preferences.

        Instructions:
        Base recommendations only on the [Retrieved Context] below.
        Include [Title], [Year], [Genre], [Rating], and a short [Summary] for each.
        If the user query is ambiguous, ask a clarifying question about genre, mood, or preferred actors.
        Do not include movies that do not fit the user's stated criteria.
        End your recommendation with a definitive closing statement. 
        Do not invite further questions or say things like "let me know if you want more options."
    """
    
    response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )
    return response.choices[0].message.content

if __name__ == "__main__":
    mood = "something intense and gripping, like a heist or crime thriller"
    
    # Retrieve from watchlist
    results = retrieve(mood)
    top_match = results["metadatas"][0][0]["title"]
    print(f"Top watchlist match: {top_match}")
    
    # Fetch from TMDB
    movies = fetch_recommendations(results)
    print(f"Found {len(movies)} movies from TMDB")
    
    # Generate recommendation
    recommendation = generate_recommendation(mood, results, movies)
    print("\n--- RECOMMENDATION ---")
    print(recommendation)