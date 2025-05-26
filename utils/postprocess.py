import pandas as pd
from typing import List, Tuple
def map_movie_id_to_title(frequent_itemsets: Tuple[set, float], movies_file_path: str = '/home/hinhnv/Hai/KDLVKP/data_mining_code/data/movielens_1m/movies.csv') -> List[List[str]]:
    """
    Map movie IDs to titles for frequent itemsets

    Args:
        movies_df: pd.DataFrame
        frequent_itemsets: Tuple[set, float]
        
    Returns:
        List[List[str]]
    """
    movies_df = pd.read_csv(movies_file_path)
    movie_id_to_title = dict(zip(movies_df['movieId'], movies_df['title']))
    new_frequent_itemsets = []
    for itemset, support in frequent_itemsets:
        new_itemset = [movie_id_to_title[item] for item in itemset]
        new_frequent_itemsets.append((tuple(new_itemset), support))
    return new_frequent_itemsets

def map_movie_id_to_title(frequent_itemsets: Tuple[set, float], movies_file_path: str = '/home/hinhnv/Hai/KDLVKP/data_mining_code/data/movielens_1m/movies.csv') -> List[List[str]]: