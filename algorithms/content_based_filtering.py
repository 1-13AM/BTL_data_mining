import numpy as np
import sys
sys.path.append('/home/hinhnv/Hai/KDLVKP/data_mining_code')
from utils.metrics import *
from utils.preprocess import filter_with_spacy_ner
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from typing import List

class ContentBasedFiltering:
    def __init__(self, user_embedding_path: str = None, item_embedding_path: str = None, user_item_interaction_path: str = None, similarity_metric: str = 'cosine', **create_embeddings_kwargs):
        
        self.load_embeddings(user_embedding_path, item_embedding_path)
        if self.item_embedding is None:
            movies_file_path, model_name, save_path = create_embeddings_kwargs.get('movies_file_path'), create_embeddings_kwargs.get('model_name'), create_embeddings_kwargs.get('item_embedding_save_path')
            self.item_embedding_map = self.create_item_embedding(movies_file_path, model_name, save_path)
            self.item_embedding = np.vstack([val for _, val in self.item_embedding_map.items()])
            
        if self.user_embedding is None:
            ratings_file_path, movie_embedding_path, save_path = create_embeddings_kwargs.get('ratings_file_path'), create_embeddings_kwargs.get('movie_embedding_path'), create_embeddings_kwargs.get('user_embedding_save_path')
            self.user_embedding_map = self.create_user_embedding(ratings_file_path, movie_embedding_path, save_path)
            self.user_embedding = np.vstack([val for _, val in self.user_embedding_map.items()])
        
        if user_item_interaction_path:
            with open(user_item_interaction_path, 'rb') as f:
                self.user_item_interaction = pkl.load(f)
        else:
            raise ValueError("user_item_interaction_path is required")
        
        self.item_keys = list(self.item_embedding_map.keys())
        self.similarity_metric = similarity_metric
        
     
    def recommend(self, user_id: int, top_k: int = 5):
        """
        recommend the top k items for the given user
        
        Args:
            user_id: int, the id of the user
            top_k: int, the number of items to recommend
            similarity_metric: str, the similarity metric to use
        """
        
        if self.similarity_metric == 'cosine':
            similarity_matrix = cosine_similarity(self.user_embedding[user_id].reshape(1, -1), self.item_embedding) # (1, n_items)
        else:
            raise ValueError(f"Invalid similarity metric: {self.similarity_metric}")
        
        # get the top k items with the highest similarity
        top_items = np.argsort(similarity_matrix)[0][::-1]

        recommended_items = []
        for i in top_items:
            item_id = self.item_keys[i]
            if item_id not in self.user_item_interaction[user_id]:
                recommended_items.append(item_id)
            if len(recommended_items) >= top_k:
                break
        return recommended_items
    
    def load_embeddings(self, user_embedding_path: str, item_embedding_path: str):
        
        self.user_embedding = None
        self.item_embedding = None
        if user_embedding_path:
            with open(user_embedding_path, 'rb') as f:
                self.user_embedding_map = pkl.load(f)
                self.user_embedding = np.vstack([val for _, val in self.user_embedding_map.items()])
        if item_embedding_path:
            with open(item_embedding_path, 'rb') as f:
                self.item_embedding_map = pkl.load(f)
                self.item_embedding = np.vstack([val for _, val in self.item_embedding_map.items()])
            
    def _create_embedding(self, tags: List[str], tag_to_embeddings: dict, dimension: int):
        embeddings = []
        for tag in tags:
            if tag in tag_to_embeddings:
                embeddings.append(tag_to_embeddings[tag])
        if len(embeddings) > 0:
            return np.mean(embeddings, axis=0)
        else:
            return np.zeros(dimension)
        
    def create_item_embedding(self,
                               movies_file_path: str, 
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
        # unique_user_tags = get_unique_tags(movie_user_tags)
        unique_genres = get_unique_tags(genres)
        
        # unique_tags = list(unique_genome_tags.union(unique_user_tags).union(unique_genres))
        unique_tags = list(unique_genome_tags.union(unique_genres))
        # filter out ner tags
        unique_tags = filter_with_spacy_ner(unique_tags)
        
        model = SentenceTransformer(model_name)
        
        print(f"Creating movie embeddings for {len(unique_tags)} tags...")
        tag_to_embeddings = {}
        for tag in tqdm(unique_tags):
            embeddings = model.encode(tag)
            tag_to_embeddings[tag] = embeddings
        
        movie_embeddings = {}
        model_dimension = model.get_sentence_embedding_dimension()
        print(f"Creating movie embeddings for {len(movies)} movies...")
        for movie_id, genome_tags, user_tags, genres in tqdm(zip(movies['movieId'], movie_genome_tags, movie_user_tags, genres)):
            
            genome_embeddings = self._create_embedding(genome_tags, tag_to_embeddings, model_dimension)
            # user_embeddings = self._create_embedding(user_tags, tag_to_embeddings, model_dimension)
            genres_embeddings = self._create_embedding(genres, tag_to_embeddings, model_dimension)
            
            denom = 0
            
            # gotta initialize the embedding for the current movie_id with a zero vector
            movie_embeddings[movie_id] = np.zeros(model.get_sentence_embedding_dimension())
            
            # for embedding in [genome_embeddings, user_embeddings, genres_embeddings]:
            for embedding in [genome_embeddings, genres_embeddings]:    
                if np.any(embedding):
                    movie_embeddings[movie_id] += embedding # otherwise this will raise an error
                    denom += 1
            if denom > 0:
                movie_embeddings[movie_id] /= denom
            
            
        if save_path:
            with open(save_path, 'wb') as f:
                pkl.dump(movie_embeddings, f)
                print(f"Movie embeddings saved to {save_path}")
            
        return movie_embeddings

    def create_user_embedding(self,
                              ratings_file_path: str = 'data/movielens_20m/rating.csv', 
                              movie_embedding_path: str = 'data/movielens_20m/movie_embedding.pkl',
                              save_path: str = 'data/movielens_20m/user_embedding.pkl'):
    
        rating = pd.read_csv(ratings_file_path)
        
        with open(movie_embedding_path, 'rb') as f:
            movie_embeddings = pkl.load(f)
            
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
                pkl.dump(user_embedding, f)
                print(f"User embeddings saved to {save_path}")
        
        return user_embedding
    
        
        
        
        
        