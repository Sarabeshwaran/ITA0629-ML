import pandas as pd
from math import log2
from collections import Counter
# Calculate entropy
def calculate_entropy(data):
    class_counts = data.value_counts(normalize=True)
    entropy = 0
    for probability in class_counts:
        entropy -= probability * log2(probability)
    return entropy

# Calculate information gain
def get_information_gain(data, target, attribute):
    entropy = calculate_entropy(data[target])
    subset_entropies = 0
    for value in data[attribute].unique():
        data_subset = data[data[attribute] == value]
        subset_entropy = calculate_entropy(data_subset[target])
        subset_entropies += (len(data_subset) / len(data)) * subset_entropy
    return entropy - subset_entropies

# Implement the ID3 algorithm
def id3(data, target, attributes):
    # Base case: All instances belong to the same class
    if len(data[target].unique()) == 1:
        return data[target].iloc[0]

    # Base case: No more attributes to split on
    if not attributes:
        return Counter(data[target]).most_common(1)[0][0]

    # Find the attribute with the highest information gain
    information_gains = {attr: get_information_gain(data, target, attr) for attr in attributes}
    best_attribute = max(information_gains, key=information_gains.get)

    # Create branch nodes for each unique value of the best attribute
    tree = {best_attribute: {}}
    for value in data[best_attribute].unique():
        data_subset = data[data[best_attribute] == value]
        attributes_subset = attributes[:]
        attributes_subset.remove(best_attribute)
        tree[best_attribute][value] = id3(data_subset, target, attributes_subset)

    return tree

# Classify a new sample
def classify_sample(tree, sample):
    node = tree
    while True:
        attribute = list(node.keys())[0]
        value = sample[attribute]
        node = node[attribute][value]
        if isinstance(node, str):  # Leaf node reached
            return node

# Print the decision tree
def print_tree(tree, indent=0):
    for attribute, branches in tree.items():
        print(f"{' ' * indent}{attribute}")
        for value, sub_tree in branches.items():
            print(f"{' ' * (indent + 1)}{value}:")
            if isinstance(sub_tree, str):
                print(f"{' ' * (indent + 2)}{sub_tree}")
            else:
                print_tree(sub_tree, indent + 2)

# Replace with your actual dataset and target attribute
data = pd.read_csv(r'C:\Users\welcome\Documents\enjoysport.csv')
target = "enjoysport"
attributes = list(data.columns)
attributes.remove(target)

tree = id3(data, target, attributes)
print(tree)
