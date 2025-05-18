import numpy as np
import sys
sys.path.append('/home/hinhnv/Hai/KDLVKP/data_mining_code')
from utils.metrics import *

class ContentBasedFiltering:
    def __init__(self, item_embedding: np.ndarray, user_embedding: np.ndarray, similarity_metric: str = 'cosine'):
        self.item_embedding = item_embedding
        self.user_embedding = user_embedding
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
        top_k_items = np.argsort(similarity_matrix)[0][::-1][:top_k]
        
        return top_k_items
        
        
        
        
        