import numpy as np
from collections import defaultdict
from itertools import combinations
import time
from .apriori import Apriori
from typing import List

class HashTreeNode:
    def __init__(self, depth: int = 1, max_leaf_size: int = 10, max_depth: int = 3):
        self.depth = depth
        self.max_leaf_size = max_leaf_size
        self.max_depth = max_depth
        self.children = {}
        self.items = []
        self.is_leaf = True
    
    def insert(self, itemset: set):
        if self.is_leaf:
            # If this is a leaf node, add the itemset
            self.items.append(itemset)
            
            # Check if we need to split the leaf
            if len(self.items) > self.max_leaf_size and self.depth < self.max_depth:
                self._split()
        else:
            # If this is an internal node, hash the itemset and insert into appropriate child
            h = self._hash(itemset)
            if h not in self.children:
                self.children[h] = HashTreeNode(self.depth + 1, self.max_leaf_size, self.max_depth)
            self.children[h].insert(itemset)
    
    def _split(self):
        self.is_leaf = False
        
        # Move items to appropriate children
        for itemset in self.items:
            h = self._hash(itemset)
            if h not in self.children:
                self.children[h] = HashTreeNode(self.depth + 1, self.max_leaf_size, self.max_depth)
            self.children[h].insert(itemset)
        self.items = []  # Clear items as they're now in children
    
    def _hash(self, itemset: set):
        sorted_itemset = sorted(list(itemset))
        return hash(sorted_itemset[self.depth - 1]) % 10

    def get_subsets(self, transaction: set, k: int) -> List[set]:
        """
        Get all k-itemsets (candidates) stored in the tree that are also 
        subsets of the given transaction.
        """
        results = []

        sorted_transaction_list = sorted(list(transaction))
        for subset_tuple in combinations(sorted_transaction_list, k):
            
            if self._contains_candidate(subset_tuple):
                # If the candidate exists in the tree, add its set version to results.
                results.append(set(subset_tuple))
        
        return results

    def _contains_candidate(self, candidate: List[int]) -> bool:
        """
        Checks if a specific, sorted candidate k-itemset exists in this
        node or its descendants.
        """
        
        if self.is_leaf:
            return set(candidate) in self.items
        else:
            if self.depth > len(candidate):
                return False
            
            hash_key = hash(candidate[self.depth - 1]) % 10 

            if hash_key in self.children:
                return self.children[hash_key]._contains_candidate(candidate)
            else:
                return False


class HashTreeApriori(Apriori):
    def __init__(self, min_support: float = 0.01, min_confidence: float = 0.5, max_leaf_size: int = 10, max_depth: int = 3, generate_rules: bool = False):
        super().__init__(min_support, min_confidence)
        self.max_leaf_size = max_leaf_size
        self.max_depth = max_depth
        self.rules = []
        self.generate_rules = generate_rules
    
    def _build_hash_tree(self, candidates: List[set]):
        """Build a hash tree from candidate itemsets"""
        candidate_length = len(candidates[0])
        tree = HashTreeNode(max_leaf_size=self.max_leaf_size, max_depth=candidate_length)
        for candidate in candidates:
            tree.insert(candidate)
        return tree
    
    def _count_supports_with_tree(self, tree: HashTreeNode, transactions: List[set], k: int):
        """Count support for candidates using hash tree"""
        supports = defaultdict(int)
        
        for transaction in transactions:
            if len(transaction) < k:
                continue
                
            # Find candidates contained in this transaction
            candidates_in_tx = tree.get_subsets(transaction, k)
            
            # Increment support count
            for candidate in candidates_in_tx:
                supports[frozenset(candidate)] += 1
        
        return {itemset: count/len(transactions) for itemset, count in supports.items()}
    
    def fit(self, transactions: List[set]):
        """Find frequent itemsets and association rules using hash tree"""
        
        # Find all unique items
        unique_items = set()
        for transaction in transactions:
            for item in transaction:
                unique_items.add(item)

        print(f"Finding frequent 1-itemsets...")
        # Find frequent 1-itemsets
        frequent_1_items = []
        for item in unique_items:
            item_set = {item}
            support = self._get_support(item_set, transactions)
            if support >= self.min_support:
                frequent_1_items.append(item_set)
                self.frequent_itemsets.append((item_set, support))
        
        print(f"Generated {len(frequent_1_items)} frequent 1-itemsets")
        print('='*20)
        # Find frequent k-itemsets for k > 1
        k = 1
        frequent_k = frequent_1_items
        while frequent_k:
            k += 1
            print(f"Finding frequent {k}-itemsets...")
            
            # convert the list of itemsets to a set of frozensets for efficiency
            frequent_k_frozensets = {frozenset(item) for item in frequent_k} 
            
            # Generate candidate k-itemsets
            candidates = []
            for i, itemset1 in enumerate(frequent_k):
                for itemset2 in frequent_k[i+1:]:
                    
                    # if the first k-2 items in the 2 itemsets are not the same, then we cannot create a candidate k-itemset
                    if k > 2 and list(itemset1)[:-1] != list(itemset2)[:-1]:
                        continue
                    
                    # Merge itemsets
                    union = itemset1.union(itemset2)
                    if len(union) == k and union not in candidates:
                        
                        all_subsets_frequent = True
                        for subset in combinations(union, k-1):
                            if frozenset(subset) not in frequent_k_frozensets:
                                all_subsets_frequent = False
                                break
                        
                        if all_subsets_frequent:
                            candidates.append(union)
            
            if not candidates:
                break
            
            print(f"Generated {len(candidates)} candidate {k}-itemsets")
            
            # Build hash tree for candidates
            hash_tree = self._build_hash_tree(candidates)
            
            # Count support for candidates using hash tree
            supports = self._count_supports_with_tree(hash_tree, transactions, k)
            
            # Filter frequent k-itemsets
            frequent_k = []
            for candidate in candidates:
                support = supports.get(frozenset(candidate), 0)
                if support >= self.min_support:
                    frequent_k.append(candidate)
                    self.frequent_itemsets.append((candidate, support))
            
            print(f"Found {len(frequent_k)} frequent {k}-itemsets")
            print('='*20)
            if self.generate_rules:
                super()._generate_rules()