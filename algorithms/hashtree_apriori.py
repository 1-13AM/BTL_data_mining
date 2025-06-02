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
        self.items = set()
        self.is_leaf = True
    
    def insert(self, itemset: frozenset):
        if self.is_leaf:
            self.items.add(itemset)
            if len(self.items) > self.max_leaf_size and self.depth < self.max_depth:
                self._split()
        else:
            h = self._hash(itemset)
            if h not in self.children:
                self.children[h] = HashTreeNode(self.depth + 1, self.max_leaf_size, self.max_depth)
            self.children[h].insert(itemset)
    
    def _split(self):
        self.is_leaf = False
        
        # Move items to appropriate children based on hash value
        for itemset in self.items:
            h = self._hash(itemset)
            if h not in self.children:
                self.children[h] = HashTreeNode(self.depth + 1, self.max_leaf_size, self.max_depth)
            self.children[h].insert(itemset)
        self.items = set()  # clear items as they're now in children nodes
    
    def _hash(self, itemset: frozenset):
        sorted_itemset = sorted(list(itemset))
        return sorted_itemset[self.depth - 1] % 10


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
            tree.insert(frozenset(candidate))
        return tree
    
    def _add_count_recursive(self, node: HashTreeNode, pick: List[int], rest: List[int], k: int, found_candidates: set):
        """
        Recursively find candidates in the hash tree that are subsets of (pick + rest)
        
        Args:
            node: Current hash tree node
            pick: Items already selected from transaction
            rest: Remaining items in transaction
            k: Target itemset length
            found_candidates: Set to store found candidates
        """
        if node.is_leaf:
            # At leaf node, check all candidates stored here
            transaction_items = set(pick + rest)
            for candidate in node.items:
                if candidate.issubset(transaction_items):
                    found_candidates.add(candidate)
            return

        n_pick = len(pick)
        n_rest = len(rest)
        n_rest_min = k - (n_pick + 1)
        if n_rest_min < 0:
            return
        n_iter = n_rest - n_rest_min
        
        for i in range(n_iter):
            curr_pick = pick + [rest[i]]
            curr_rest = rest[i+1:]
            
            # Hash based on the item at the current node's depth position
            # if len(curr_pick) > node.depth - 1:
            hash_item = curr_pick[node.depth - 1]
            key = hash_item % 10
            if key in node.children:
                self._add_count_recursive(node.children[key], curr_pick, curr_rest, k, found_candidates)
    
    def _count_supports_with_tree(self, tree: HashTreeNode, transactions: List[set], k: int):
        """Count support for candidates using hash tree"""
        supports = defaultdict(int)
        
        for transaction in transactions:
            if len(transaction) < k:
                continue
            
            transaction_list = sorted(list(transaction))
            found_candidates = set()
            self._add_count_recursive(tree, [], transaction_list, k, found_candidates)
            # Increment support count for found candidates
            for candidate in found_candidates:
                supports[candidate] += 1
        
        
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
            frequent_k_frozensets = {frozenset(item) for item in frequent_k} 
            candidates = []
            for i, itemset1 in enumerate(frequent_k):
                for itemset2 in frequent_k[i+1:]:
    
                    # if the first k-2 items in the 2 itemsets are not the same, then we cannot create a candidate k-itemset
                    if k > 2 and list(itemset1)[:-1] != list(itemset2)[:-1]:
                        continue
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
            
            # build hash tree
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