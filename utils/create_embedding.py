from tqdm import tqdm
import numpy as np
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List
import spacy

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