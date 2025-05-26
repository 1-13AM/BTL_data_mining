import numpy as np
from collections import defaultdict, Counter
from itertools import combinations
import time
import pickle

class FPNode:
    def __init__(self, item=None, count=0, parent=None):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.node_link = None  # Link to next node with same item
    
    def increase_count(self, count=1):
        self.count += count
    
    def display(self, ind=1):
        print('  ' * ind, self.item, ' ', self.count)
        for child in self.children.values():
            child.display(ind + 1)

class FPTree:
    def __init__(self, transactions, min_support_count):
        self.min_support_count = min_support_count
        self.root = FPNode()
        self.header_table = {}
        self.frequent_items = []
        
        # count item occurences
        item_counts = Counter()
        for transaction in transactions:
            for item in transaction:
                item_counts[item] += 1
        
        # filter frequent items and sort by frequency & in descending order
        self.frequent_items = [item for item, count in item_counts.items() if count >= min_support_count]
        self.frequent_items.sort(key=lambda x: item_counts[x], reverse=True)

        for item in self.frequent_items:
            self.header_table[item] = None
        
        # Build FP-tree
        for transaction in transactions:
            # Filter and sort transaction items by frequency
            filtered_items = [item for item in transaction if item in self.frequent_items]
            filtered_items.sort(key=lambda x: item_counts[x], reverse=True)
            
            if filtered_items:
                self._insert_tree(filtered_items, self.root)
    
    def _insert_tree(self, items, node):
        if not items:
            return
            
        first = items[0]
        child = node.children.get(first)
        
        if child is not None:
            child.increase_count()
        else:
            child = FPNode(first, 1, node)
            node.children[first] = child
            self._update_header(child)
        
        remaining_items = items[1:]
        if remaining_items:
            self._insert_tree(remaining_items, child)
    
    def _update_header(self, node):
        if self.header_table[node.item] is None:
            self.header_table[node.item] = node
        else:
            current = self.header_table[node.item]
            while current.node_link is not None:
                current = current.node_link
            current.node_link = node
    
    def _get_prefix_paths(self, item):
        """Get all prefix paths for a given item"""
        paths = []
        node = self.header_table.get(item)
        
        while node is not None:
            path = []
            parent = node.parent
            while parent is not None and parent.item is not None:
                path.append(parent.item)
                parent = parent.parent
            
            if path:
                paths.append((path[::-1], node.count))  # Reverse to get correct order
            node = node.node_link
        
        return paths

class FPGrowth:
    def __init__(self, min_support=0.01, min_confidence=0.5, generate_rules=True):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.generate_rules = generate_rules
        self.frequent_itemsets = []
        self.rules = []
        self.num_transactions = 0
    
    def _get_support(self, itemset: set, transactions: list) -> float:
        """Calculate support for an itemset"""
        count = 0
        for transaction in transactions:
            if all(item in transaction for item in itemset):
                count += 1
        return count / len(transactions)
    
    def _mine_fp_tree(self, fp_tree, alpha, frequent_itemsets, min_support_count):
        """Mine frequent patterns from FP-tree using iterative approach"""
        
        # Process items in reverse order of frequency
        for item in reversed(fp_tree.frequent_items):
            # Create new frequent itemset
            new_itemset = alpha | {item}
            
            # Calculate support for this item
            support_count = 0
            node = fp_tree.header_table.get(item)
            while node is not None:
                support_count += node.count
                node = node.node_link
            
            support = support_count / self.num_transactions
            
            if support >= self.min_support:
                frequent_itemsets.append((new_itemset, support))
            
            # Get conditional pattern base
            prefix_paths = fp_tree._get_prefix_paths(item)
            
            if prefix_paths:
                # Create conditional transactions
                conditional_transactions = []
                for path, count in prefix_paths:
                    for _ in range(count):
                        conditional_transactions.append(path)
                
                # Build conditional FP-tree only if we have enough transactions
                if len(conditional_transactions) >= min_support_count:
                    conditional_tree = FPTree(conditional_transactions, min_support_count)
                    
                    # Only recurse if the conditional tree has frequent items
                    if conditional_tree.frequent_items:
                        self._mine_fp_tree(conditional_tree, new_itemset, frequent_itemsets, min_support_count)
    
    def fit(self, transactions: list):
        """Find frequent itemsets and association rules using FP-Growth"""
        self.num_transactions = len(transactions)
        min_support_count = int(self.min_support * self.num_transactions)
        
        # Build initial FP-tree
        fp_tree = FPTree(transactions, min_support_count)
        
        print(f"Found {len(fp_tree.frequent_items)} frequent items")

        # Mine frequent patterns
        frequent_itemsets = []
        self._mine_fp_tree(fp_tree, set(), frequent_itemsets, min_support_count)
        self.frequent_itemsets = frequent_itemsets
        
        print(f"Found {len(self.frequent_itemsets)} frequent itemsets")
        
        if self.generate_rules:
            self._generate_rules()
    
    def _generate_rules(self):
        """Generate association rules from frequent itemsets"""
        print(f"Generating association rules...")
        
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
    def load_results(cls, filepath: str) -> 'FPGrowth':
        """Load frequent itemsets and rules from a file"""
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
        fp_growth = cls(results['min_support'], results['min_confidence'])
        fp_growth.frequent_itemsets = results['frequent_itemsets']
        fp_growth.rules = results['rules']
        
        print(f"Loaded {len(fp_growth.frequent_itemsets)} frequent itemsets and {len(fp_growth.rules)} rules")
        
        return fp_growth