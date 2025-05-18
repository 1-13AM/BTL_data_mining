from ..algorithms.apriori import Apriori
import pickle as pkl

with open('data/movielens_1m/transaction.pkl', 'rb') as f:
    transaction_list = pkl.load(f)

apriori = Apriori(min_support=0.15, min_confidence=0.5)
apriori.fit(transaction_list)
apriori.save_results('data/movielens_1m/apriori_results.pkl')