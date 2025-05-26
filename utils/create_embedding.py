from tqdm import tqdm
import numpy as np
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List
import spacy

def create_movie_embedding(movies_file_path: str, 
                           model_name: str = 'all-MiniLM-L6-v2', 
                           save_path: str = '/data/movielens_20m/movie_embedding.pkl'):
    movies = pd.read_csv(movies_file_path)
    movie_genome_tags = movies['genome_tags'].apply(lambda x: x.split('|') if isinstance(x, str) else [])
    movie_user_tags = movies['user_tags'].apply(lambda x: x.split('|') if isinstance(x, str) else [])
    genres = movies['genres'].apply(lambda x: x.split('|') if isinstance(x, str) else [])
    
    def get_unique_tags(tags_list):
        unique_tags = set()
        for tags in tags_list:
            unique_tags.update([tag for tag in tags])
        return unique_tags
    
    unique_genome_tags = get_unique_tags(movie_genome_tags)
    unique_user_tags = get_unique_tags(movie_user_tags)
    unique_genres = get_unique_tags(genres)
    
    unique_tags = list(unique_genome_tags.union(unique_user_tags).union(unique_genres))
    model = SentenceTransformer(model_name)
    
    print(f"Creating movie embeddings for {len(unique_tags)} tags...")
    tag_to_embeddings = {}
    for tag in tqdm(unique_tags):
        embeddings = model.encode(tag)
        tag_to_embeddings[tag] = embeddings
    
    movie_embeddings = {}
    
    print(f"Creating movie embeddings for {len(movies)} movies...")
    for movie_id, genome_tags, user_tags, genres in tqdm(zip(movies['movieId'], movie_genome_tags, movie_user_tags, genres)):
        genome_embeddings = np.mean([tag_to_embeddings[tag] for tag in genome_tags], axis=0) if len(genome_tags) > 0 else np.zeros(model.get_sentence_embedding_dimension())
        user_embeddings = np.mean([tag_to_embeddings[tag] for tag in user_tags], axis=0) if len(user_tags) > 0 else np.zeros(model.get_sentence_embedding_dimension())
        genres_embeddings = np.mean([tag_to_embeddings[tag] for tag in genres], axis=0) if len(genres) > 0 else np.zeros(model.get_sentence_embedding_dimension())
        
        denom = 0
        
        # gotta initialize the embedding for the current movie_id with a zero vector
        movie_embeddings[movie_id] = np.zeros(model.get_sentence_embedding_dimension())
        
        for embedding in [genome_embeddings, user_embeddings, genres_embeddings]:
            if np.any(embedding):
                movie_embeddings[movie_id] += embedding # otherwise this will raise an error
                denom += 1
        movie_embeddings[movie_id] /= denom
        
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(movie_embeddings, f)
            print(f"Movie embeddings saved to {save_path}")
        
    return movie_embeddings

def create_user_embedding(ratings_file_path: str = 'data/movielens_20m/rating.csv', 
                          movie_embedding_path: str = 'data/movielens_20m/movie_embedding.pkl',
                          save_path: str = 'data/movielens_20m/user_embedding.pkl'):
    
    rating = pd.read_csv(ratings_file_path)
    
    with open(movie_embedding_path, 'rb') as f:
        movie_embeddings = pickle.load(f)
        
    user_embedding = {}
    unique_user_ids = [num.item() for num in rating['userId'].unique()]
    
    _embedding_shape = movie_embeddings[1].shape
    
    for user_id in tqdm(unique_user_ids):
        # Initialize the embedding for the current user_id with a zero vector
        user_embedding[user_id] = np.zeros(_embedding_shape) 
        denom = 0

        user_ratings_for_current_user = rating[rating['userId'] == user_id]

        for _, rated_row in user_ratings_for_current_user.iterrows():
            movie_id = rated_row['movieId']
            actual_rating = rated_row['rating']
            
            current_movie_embedding = movie_embeddings[movie_id]
            
            if np.any(current_movie_embedding): 
                denom += 1
                user_embedding[user_id] += actual_rating * current_movie_embedding
                
        if denom > 0:
            user_embedding[user_id] /= denom

    print(f"Finished processing embeddings for {len(user_embedding)} users.")
    
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(user_embedding, f)
            print(f"User embeddings saved to {save_path}")
    
    return user_embedding

def filter_with_spacy_ner(tags: List[str]) -> List[str]:
    """
    Remove named entities using spaCy NER
    
    Parameters:
    -----------
    tags : List[str]
        List of tags to filter
        
    Returns:
    --------
    List[str]
        Filtered tags without named entities
    """
    nlp = spacy.load("en_core_web_sm")
    if nlp is None:
        return tags
    
    filtered_tags = []
    
    for tag in tags:
        doc = nlp(tag)
        
        # Check if the tag contains named entities
        has_named_entity = False
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'WORK_OF_ART', 'EVENT', 'FAC', 'NORP']:
                has_named_entity = True
                break
        
        if not has_named_entity:
            words = tag.split()
            if len(words) >= 2 and all(word[0].isupper() for word in words if word):
                has_named_entity = True
        
        if not has_named_entity:
            filtered_tags.append(tag)
    
    return filtered_tags