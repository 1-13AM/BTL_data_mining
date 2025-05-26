import numpy as np
from collections import defaultdict
from itertools import combinations
import time
import pickle

class Apriori:
    def __init__(self, min_support=0.01, min_confidence=0.5, generate_rules=False):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.generate_rules = generate_rules
        self.frequent_itemsets = []
        self.rules = []
    
    def _get_support(self, itemset: set, transactions: list) -> float:
        """Calculate support for an itemset"""
        count = 0
        for transaction in transactions:
            if all(item in transaction for item in itemset):
                count += 1
        return count / len(transactions)
    
    def fit(self, transactions: list) -> 'Apriori':
        """Find frequent itemsets and association rules"""
        # Find all unique items
        unique_items = set()
        for transaction in transactions:
            for item in transaction:
                unique_items.add(item)
        
        # Find frequent 1-itemsets
        print(f"Generating 1-itemsets")
        frequent_1_items = []
        for item in unique_items:
            item_set = {item}
            support = self._get_support(item_set, transactions)
            if support >= self.min_support:
                frequent_1_items.append(item_set)
                self.frequent_itemsets.append((item_set, support))
        
        # find frequent k-itemsets for k > 1
        k = 1
        frequent_k = frequent_1_items
        
        print(f"Generated {len(frequent_1_items)} 1-itemsets")
        print('='*20)
        while frequent_k:
            k += 1
            print(f"Generating {k}-itemsets")
            # Generate candidate k-itemsets
            candidates = []
            for i, itemset1 in enumerate(frequent_k):
                for itemset2 in frequent_k[i+1:]:
                    items1 = sorted(itemset1)
                    items2 = sorted(itemset2)
                    
                    # if the first k-2 items are different, they won't make a k-itemset candidate
                    # so we skip
                    if k > 2 and items1[:-1] != items2[:-1]:
                        continue
                    
                    # merge itemsets
                    union = itemset1.union(itemset2)
                    if len(union) == k:
                        # this is to ensure that if a candidate k-itemset is generated, all of its (k-1)-subsets must be frequent
                        all_subsets_frequent = True
                        for subset in combinations(union, k-1):
                            subset_set = set(subset)
                            if subset_set not in frequent_k:
                                all_subsets_frequent = False
                                break
                        
                        if all_subsets_frequent:
                            candidates.append(union)
            
            # find frequent k-itemsets
            frequent_k = []
            num_k_frequent_itemsets = 0
            for candidate in candidates:
                support = self._get_support(candidate, transactions)
                if support >= self.min_support:
                    frequent_k.append(candidate)
                    self.frequent_itemsets.append((candidate, support))
                    num_k_frequent_itemsets += 1
            
            print(f"Generated {num_k_frequent_itemsets} frequent {k}-itemsets")
            print('='*20)
        
        if self.generate_rules:
            self._generate_rules()
    
    def _generate_rules(self):
        # Generate association rules
        for itemset, support in self.frequent_itemsets:
            if len(itemset) > 1:
                for i in range(1, len(itemset)):
                    for conditional_items in combinations(itemset, i):
                        condition = set(conditional_items)
                        consequent = itemset - condition
                        
                        # Find support of condition
                        condition_support = None
                        for item_set, supp in self.frequent_itemsets:
                            if item_set == condition:
                                condition_support = supp
                                break
                        
                        if condition_support:
                            # Calculate confidence
                            confidence = support / condition_support
                            
                            if confidence >= self.min_confidence:
                                self.rules.append((condition, consequent, support, confidence))
        
        print(f"Generated {len(self.rules)} association rules")
    
    def get_frequent_itemsets(self) -> list:
        """Return frequent itemsets sorted by support"""
        return sorted(self.frequent_itemsets, key=lambda x: x[1], reverse=True)
    
    def get_rules(self) -> list:
        """Return association rules sorted by confidence"""
        return sorted(self.rules, key=lambda x: x[3], reverse=True)
    
    def save_results(self, filepath):
        """Save frequent itemsets and rules to a file"""
        results = {
            'min_support': self.min_support,
            'min_confidence': self.min_confidence,
            'frequent_itemsets': self.frequent_itemsets,
            'rules': self.rules
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Results saved to {filepath}")
    
    @classmethod
    def load_results(cls, filepath: str) -> 'Apriori':
        """Load frequent itemsets and rules from a file"""
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
        apriori = cls(results['min_support'], results['min_confidence'])
        apriori.frequent_itemsets = results['frequent_itemsets']
        apriori.rules = results['rules']
        
        print(f"Loaded {len(cls.frequent_itemsets)} frequent itemsets and {len(cls.rules)} rules")
        
        return cls
