# BÀI TẬP - THUẬT TOÁN ID3

import math
import pandas as pd


# 1. TẠO DỮ LIỆU

data = [
    ["<=30", "high", "no", "fair", "no"],
    ["<=30", "high", "no", "excellent", "no"],
    ["31...40", "high", "no", "fair", "yes"],
    [">40", "medium", "no", "fair", "yes"],
    [">40", "low", "yes", "fair", "yes"],
    [">40", "low", "yes", "excellent", "no"],
    ["31...40", "low", "yes", "excellent", "yes"],
    ["<=30", "medium", "no", "fair", "no"],
    ["<=30", "low", "yes", "fair", "yes"],
    [">40", "medium", "yes", "fair", "yes"],
    ["<=30", "medium", "yes", "excellent", "yes"],
    ["31...40", "medium", "no", "excellent", "yes"],
    ["31...40", "high", "yes", "fair", "yes"],
    [">40", "medium", "no", "excellent", "no"]
]

columns = [
    "age",
    "income",
    "student",
    "credit_rating",
    "buys_computer"
]

df = pd.DataFrame(data, columns=columns)


# 2. TÍNH ENTROPY

def entropy(data):

    labels = data.iloc[:, -1]

    counts = labels.value_counts()

    total = len(labels)

    result = 0

    for count in counts:
        probability = count / total
        result -= probability * math.log2(probability)

    return result


# 3. TÍNH INFORMATION GAIN

def information_gain(data, attribute):

    total_entropy = entropy(data)

    values = data[attribute].unique()

    weighted_entropy = 0

    for value in values:

        subset = data[data[attribute] == value]

        weight = len(subset) / len(data)

        weighted_entropy += weight * entropy(subset)

    gain = total_entropy - weighted_entropy

    return gain


# 4. CHỌN THUỘC TÍNH TỐT NHẤT

def best_attribute(data, attributes):

    gains = {}

    for attribute in attributes:

        gains[attribute] = information_gain(
            data,
            attribute
        )

    best = max(gains, key=gains.get)

    return best, gains


# 5. THUẬT TOÁN ID3

def ID3(data, attributes):

    labels = data.iloc[:, -1]

    if len(labels.unique()) == 1:
        return labels.iloc[0]

    if len(attributes) == 0:
        return labels.mode()[0]

    best, gains = best_attribute(
        data,
        attributes
    )

    tree = {
        best: {}
    }

    remaining_attributes = [
        attribute
        for attribute in attributes
        if attribute != best
    ]

    for value in data[best].unique():

        subset = data[
            data[best] == value
        ]

        if len(subset) == 0:
            tree[best][value] = labels.mode()[0]

        else:
            tree[best][value] = ID3(
                subset,
                remaining_attributes
            )

    return tree


# 6. XÂY DỰNG CÂY

attributes = [
    "age",
    "income",
    "student",
    "credit_rating"
]

tree = ID3(
    df,
    attributes
)


# 7. IN INFORMATION GAIN

print("INFORMATION GAIN BAN ĐẦU")

for attribute in attributes:

    gain = information_gain(
        df,
        attribute
    )

    print(
        attribute,
        "=",
        round(gain, 4)
    )


# 8. IN CÂY QUYẾT ĐỊNH

print("\nCÂY QUYẾT ĐỊNH ID3")

print(tree)