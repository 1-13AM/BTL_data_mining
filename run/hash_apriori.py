from ..algorithms.hash_tree_apriori import HashTreeApriori
import pickle as pkl

with open('data/movielens_1m/transaction.pkl', 'rb') as f:
    transaction_list = pkl.load(f)

hash_tree_apriori = HashTreeApriori(min_support=0.3, min_confidence=0.5, max_leaf_size=64, max_depth=4)
hash_tree_apriori.fit(transaction_list)
hash_tree_apriori.save_results('data/movielens_1m/hash_tree_apriori_results.pkl')