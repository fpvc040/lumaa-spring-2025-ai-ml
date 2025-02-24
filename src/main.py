import pandas as pd
import argparse
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import ast

# Load sentence transformer models
model = SentenceTransformer("all-MiniLM-L6-v2")
metadata_model = SentenceTransformer("paraphrase-distilroberta-base-v1")

def load_and_align_data(movies_file, metadata_file):
    """Load movies and metadata, filter to only common movie IDs, and return as Pandas DataFrames."""
    
    metadata_df = pd.read_csv(metadata_file, delimiter="\t", header=None, dtype={0: str})
    metadata_df.columns = [
        'Wikipedia Movie ID', 'Freebase Movie ID', 'Movie Name', 'Release Date',
        'Box Office Revenue', 'Runtime', 'Languages', 'Countries', 'Genres'
    ]
    movies_df = pd.read_csv(movies_file, delimiter="\t", header=None, names=["Wikipedia Movie ID", "Plot"], dtype={"Wikipedia Movie ID": str})
    merged_df = movies_df.merge(metadata_df, on="Wikipedia Movie ID", how="inner")
    aligned_movies_df = merged_df[['Wikipedia Movie ID', 'Plot']]
    
    return aligned_movies_df, merged_df

def preprocess_plots(movie_plots):
    """Encodes movie plots into embeddings and saves as a NumPy file for faster reloading."""
    if os.path.exists("embeddings.npz"):
        data = np.load("embeddings.npz", allow_pickle=True)
        return data["movie_plots"], data["embeddings"]
    
    movie_plots = movie_plots.to_numpy()
    embeddings = model.encode(movie_plots[:, 1], batch_size=32, show_progress_bar=True)
    np.savez("embeddings.npz", movie_plots=movie_plots, embeddings=embeddings)
    return movie_plots, embeddings

def preprocess_metadata_embeddings(metadata_dict):
    """Encode metadata (Genres, Languages, Countries) into embeddings and save them."""
    if os.path.exists("metadata_embeddings.npz"):
        return np.load("metadata_embeddings.npz")['metadata_embeddings']
    
    metadata_texts = []
    for metadata in metadata_dict.values():

        genres_dict = ast.literal_eval(metadata.get('Genres', '{}'))
        languages_dict = ast.literal_eval(metadata.get('Languages', '{}'))
        countries_dict = ast.literal_eval(metadata.get('Countries', '{}'))

        # Extract genre, language, and country names
        genres = ", ".join([genre[1] for genre in genres_dict.items() if len(genre) > 1]).encode('utf-8', 'ignore').decode('utf-8')
        languages = ", ".join([lang[1] for lang in languages_dict.items() if len(lang) > 1]).encode('utf-8', 'ignore').decode('utf-8')
        countries = ", ".join([country[1] for country in countries_dict.items() if len(country) > 1]).encode('utf-8', 'ignore').decode('utf-8')
        metadata_texts.append(f"Genres: {genres}, Languages: {languages}, Countries: {countries}")
    
    metadata_embeddings = metadata_model.encode(metadata_texts, batch_size=32, show_progress_bar=True)
    np.savez("metadata_embeddings.npz", metadata_embeddings=metadata_embeddings)
    return metadata_embeddings

def find_closest_match(user_input, plot_embeddings, metadata_embeddings, metadata_dict, num_recommendations=3):
    """Find the closest movie embeddings to the user query and return top N recommendations."""
    query_plot_embedding = model.encode([user_input])
    query_metadata_embedding = metadata_model.encode([user_input])
    
    plot_similarities = cosine_similarity(query_plot_embedding, plot_embeddings)
    metadata_similarities = cosine_similarity(query_metadata_embedding, metadata_embeddings)
    combined_similarities = (plot_similarities + metadata_similarities) / 2
    
    movie_ids = np.array(list(metadata_dict.keys()))
    recommendations = []

    
    for idx in np.argsort(combined_similarities[0])[::-1]:
        movie_id = movie_ids[idx]
        movie_data = metadata_dict.get(movie_id, {})
        recommendations.append({
            'Movie ID': movie_id,
            'Movie Name': movie_data.get('Movie Name', 'Unknown'),
            'Similarity': combined_similarities[0][idx],
            'Metadata': movie_data
        })
        if len(recommendations) == num_recommendations:
            break
    
    return recommendations

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Movie Recommendation System")
    parser.add_argument("--file", help="Path to movie data", default="../MovieSummaries/plot_summaries.txt")
    parser.add_argument("--metadata", help="Path to movie metadata", default="../MovieSummaries/movie.metadata.tsv")
    parser.add_argument("--num_recs", default=5, type=int)
    parser.add_argument("--query", help="User input query for recommendations", required=True, type=str)
    args = parser.parse_args()
    
    print("Loading and processing data...")
    movies_plot_df, metadata_df = load_and_align_data(args.file, args.metadata)
    movies, plot_embeddings = preprocess_plots(movies_plot_df)
    
    movies_metadata_dict = {row['Wikipedia Movie ID']: row.to_dict() for _, row in metadata_df.iterrows()}
    metadata_embeddings = preprocess_metadata_embeddings(movies_metadata_dict)
    
    print("Finding recommendations...")
    recommendations = find_closest_match(args.query, plot_embeddings, metadata_embeddings, movies_metadata_dict, num_recommendations=args.num_recs)
    
    for rec in recommendations:
        print(f"Movie ID: {rec['Movie ID']}")
        print(f"Movie Name: {rec['Movie Name']}")
        print(f"Similarity: {rec['Similarity']:.4f}")
        print(f"Metadata: {rec['Metadata']}")
        print("-" * 50)
