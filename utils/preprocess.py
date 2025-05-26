# contains functions for data preprocessing
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
import spacy
from collections import defaultdict
import os
import pickle
import time

def concatenate_movie_tags(
    movies_file:str='data/movie.csv',
    user_tags_file:str='data/tag.csv',
    genome_scores_file:str='data/genome_scores.csv',
    genome_tags_file:str='data/genome_tags.csv',
    min_relevance:float=0.2
) -> pd.DataFrame:
    """
    Concatenate user tags and genome tags to movie data.
    
    Parameters:
    -----------
    movies_file : str
        Path to movie.csv file
    user_tags_file : str
        Path to tag.csv file containing user-provided tags
    genome_scores_file : str
        Path to genome_scores.csv file
    genome_tags_file : str
        Path to genome_tags.csv file
    min_relevance : float
        Minimum relevance score to include a genome tag (default: 0.2)
        
    Returns:
    --------
    pandas.DataFrame
        Movies dataframe with concatenated tag columns
    """
    # Load DataFrames
    movies_df = pd.read_csv(movies_file)
    user_tags_df = pd.read_csv(user_tags_file)
    genome_scores_df = pd.read_csv(genome_scores_file)
    genome_tags_info_df = pd.read_csv(genome_tags_file)
    
    # process user tags
    user_tags_aggregated = (user_tags_df.groupby('movieId')['tag']
                           .apply(lambda x: '|'.join(x.astype(str)))
                           .reset_index()
                           .rename(columns={'tag': 'user_tags'}))
    
    # process genome tags
    genome_tags_merged = pd.merge(genome_scores_df, genome_tags_info_df, on='tagId')
    genome_tags_filtered = genome_tags_merged[genome_tags_merged['relevance'] >= min_relevance]
    genome_tags_aggregated = (genome_tags_filtered.groupby('movieId')['tag']
                             .apply(lambda x: '|'.join(x.astype(str)))
                             .reset_index()
                             .rename(columns={'tag': 'genome_tags'}))
    
    # merge tag columns into movie dataframe
    result_df = (movies_df
                .merge(user_tags_aggregated, on='movieId', how='left')
                .merge(genome_tags_aggregated, on='movieId', how='left'))
    
    # fill nan values with empty strings
    result_df['user_tags'] = result_df['user_tags'].fillna('')
    result_df['genome_tags'] = result_df['genome_tags'].fillna('')
    
    return result_df

def summarize_missing_values(file_paths:dict[str, str]) -> pd.DataFrame:
    """
    Create a summary table of missing values for each file.
    
    Parameters:
    -----------
    file_paths : dict
        Dictionary mapping file names to their file paths
        
    Returns:
    --------
    pandas.DataFrame
        Summary table of missing values
    """
    summary_rows = []
    
    for name, path in file_paths.items():
        # load the dataframe
        df = pd.read_csv(path)
        
        # calculate overall stats
        total_rows = len(df)
        total_cols = len(df.columns)
        total_cells = total_rows * total_cols
        missing_cells = df.isna().sum().sum()
        
        # calculate per-column missing values
        col_missing = df.isna().sum()
        col_missing_pct = col_missing / total_rows * 100
        
        # column details formatted as strings
        column_details = [f"{col}: {miss}/{total_rows} ({pct:.1f}%)" 
                         for col, miss, pct in zip(col_missing.index, 
                                                 col_missing.values, 
                                                 col_missing_pct.values) 
                         if miss > 0]
        
        if not column_details:
            column_details = ["No missing values"]
            
        # Create summary row
        summary_rows.append({
            'File': name,
            'Total Rows': total_rows,
            'Total Columns': total_cols,
            'Missing Cells': f"{missing_cells}/{total_cells} ({missing_cells/total_cells*100:.1f}%)",
            'Columns with Missing Values': len([c for c in col_missing.values if c > 0]),
            'Details': '\n'.join(column_details)
        })
    
    summary_df = pd.DataFrame(summary_rows)
    
    return summary_df

def create_user_item_matrix(
    ratings_file:str='data/ratings.csv'
) -> np.ndarray:
    """
    return a user-item matrix with shape (num_users, num_items)
    """
    ratings = pd.read_csv(ratings_file, nrows=1000000)
    
    
    user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')
    return user_item_matrix.to_numpy()

def create_transactions(file_path, rating_threshold=3.5, chunk_size=100000, save_path=None):
    """
    Convert movie ratings into transaction format for association rule mining
    
    Parameters:
    - file_path: Path to the rating.csv file
    - rating_threshold: Minimum rating to consider a movie as "liked" (default: 3.5)
    - save_path: Optional path to save the transactions for reuse
    
    Returns:
    - List of sets, where each set contains movie IDs that a user liked
    """
    print(f"Loading transactions from {file_path} with threshold {rating_threshold}...")
    
    # Check if saved transactions exist
    if save_path and os.path.exists(save_path):
        print(f"Loading pre-processed transactions from {save_path}...")
        with open(save_path, 'rb') as f:
            transaction_list = pickle.load(f)
        print(f"Loaded {len(transaction_list)} transactions")
        return transaction_list
    
    # read data in chunks
    chunks = pd.read_csv(file_path, chunksize=chunk_size)
    
    transactions = defaultdict(set)
    total_ratings = 0
    
    for i, chunk in enumerate(chunks):
        if i >= 1:
            break
        print(f"Processing chunk {i+1}...")
        # only consider the ratings above threshold
        filtered_chunk = chunk[chunk['rating'] >= rating_threshold]
        
        for _, row in filtered_chunk.iterrows():
            try:
                transactions[row['userId']].add(row['movieId'].item())
            except:
               transactions[row['userId']].add(row['movieId'])
            
        total_ratings += len(chunk)
    
    transaction_list = list(transactions.values())
    
    transaction_list = [sorted(t) for t in transaction_list if len(t) > 0]
    
    print(f"Created {len(transaction_list)} transactions from {total_ratings} ratings")
    
    # Save transactions if path is provided
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        print(f"Saving transactions to {save_path}...")
        with open(save_path, 'wb') as f:
            pickle.dump(transaction_list, f)
    
    return transaction_list

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

def embed_text(model:SentenceTransformer, text:str) -> np.ndarray:
    return model.encode(text)
    
    
    
