import numpy as np
import pickle as pkl
import sys
sys.path.append('/home/hinhnv/Hai/KDLVKP/data_mining_code')
from utils.metrics import pearson_correlation, cosine_similarity
# from ..utils.metrics import pearson_correlation, cosine_similarity

class ColabFiltering:
    def __init__(self, strategy: str = 'user_based', similarity_metric: str = 'cosine', k: int = 5):
        self.strategy = strategy
        self.similarity_metric = similarity_metric
        self.k = k
        self.user_item_matrix = None
        self.similarity_matrix = None
        self.mask_matrix = None
        self.top_k_items_list = []
    
    # def fit(self, user_item_matrix: np.ndarray):
        
    #     self.user_item_matrix = user_item_matrix
    #     self.mask_matrix = np.isnan(user_item_matrix)
        
    #     if self.strategy == 'user_based':
    #         self.similarity_matrix = self._calculate_similarity(user_item_matrix, user_item_matrix)
    #     else:
    #         self.similarity_matrix = self._calculate_similarity(user_item_matrix.T, user_item_matrix.T)
        
    #     mean_rating = self.user_item_matrix[~np.isnan(self.user_item_matrix)].mean()
    #     for i in range(self.similarity_matrix.shape[0]):
    #         top_ks = np.argsort(self.similarity_matrix[i])[::-1][:self.k]
            
    #         user_item_matrix_copy = user_item_matrix.copy()
    #         if self.strategy == 'item_based':
    #             user_item_matrix_copy = user_item_matrix_copy.T
            
    #         for j in range(user_item_matrix_copy.shape[1]):
    #             nom, denom = 0, 0
    #             for k in top_ks:
    #                 if user_item_matrix_copy[k, j] != np.nan:
    #                     nom += self.similarity_matrix[i, k] * user_item_matrix_copy[k, j]
    #                     denom += np.abs(self.similarity_matrix[i, k])
                
    #             if denom != 0:
    #                 if self.strategy == 'user_based':
    #                     self.user_item_matrix[i, j] = nom / denom
    #                 else:
    #                     self.user_item_matrix[j, i] = nom / denom
            
    #         self.user_item_matrix = np.nan_to_num(self.user_item_matrix, mean_rating)
    
    # an optimized version of fit
    def fit(self, user_item_matrix: np.ndarray):
        
        self.user_item_matrix = user_item_matrix.copy()
        # precompute mask & zero-filled version
        mask_mat = ~np.isnan(self.user_item_matrix)
        uim0 = np.nan_to_num(self.user_item_matrix, 0.0)
        mean_rating = np.nanmean(self.user_item_matrix)

        if self.strategy=='item_based':
            uim = uim0.T
            mask = mask_mat.T
        else:
            uim = uim0
            mask = mask_mat

        sim = self._calculate_similarity(uim, uim)
        self.similarity_matrix = sim

        n = sim.shape[0]
        for i in range(n):
            row = sim[i].copy()
            row[i] = -np.inf
            top_ks = np.argsort(row)[-self.k:]
            
            Rk   = uim[top_ks, :]            # (k, n_items)
            mk   = mask[top_ks, :]           # (k, n_items)
            w    = sim[i, top_ks][:, None]    # (k,1)
            nom  = np.nansum(w * Rk, axis=0)
            denom  = np.sum(np.abs(w) * mk, axis=0)
            # pred = np.where(denom>0, nom/denom, mean_rating)
            pred = np.where(denom>0, nom/denom, -1)
            if self.strategy=='user_based':
                missing = ~mask_mat[i, :]         
                self.user_item_matrix[i, missing] = pred[missing]
            else:
                missing = ~mask_mat[:, i]              
                self.user_item_matrix[missing, i] = pred[missing]
    
    def _calculate_similarity(self, matrix_1:np.ndarray, matrix_2:np.ndarray):
        if self.similarity_metric == 'cosine':
            return cosine_similarity(matrix_1, matrix_2)
        elif self.similarity_metric == 'pearson':
            return pearson_correlation(matrix_1, matrix_2)
    
    def recommend(self, user_id:int, top_k:int = 5):
        """
        recommend the top k items for the given user
        """
        
        return np.argsort(np.where(self.mask_matrix[user_id], self.user_item_matrix[user_id], -1))[::-1][:top_k]
        
    def save_results(self, filepath:str):
        """
        save the results to the given filepath
        """
        with open(filepath, 'wb') as f:
            pkl.dump({
                'strategy': self.strategy,
                'similarity_metric': self.similarity_metric,
                'k': self.k,
                'user_item_matrix': self.user_item_matrix,
                'similarity_matrix': self.similarity_matrix,
                'mask_matrix': self.mask_matrix,
                'top_k_items_list': self.top_k_items_list
            }, f)
    
    @classmethod
    def load_results(cls, filepath:str):
        with open(filepath, 'rb') as f:
            results = pkl.load(f)
            
        instance = cls(results['strategy'], results['similarity_metric'], results['k'])
        instance.user_item_matrix = results['user_item_matrix']
        instance.similarity_matrix = results['similarity_matrix']
        instance.mask_matrix = results['mask_matrix']
        instance.top_k_items_list = results['top_k_items_list']
        return instance
    
    
        