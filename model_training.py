import csv
import math

def load_data(filename, limit=1000):
    """Loads CSV data using only built-in modules."""
    data = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            # Convert strings to floats
            row_float = [float(x) for x in row]
            data.append(row_float)
            if i >= limit: # Limiting for speed, as pure python is slower
                break
    return data

def euclidean_distance(v1, v2):
    """Calculates distance between two ECG signals (first 187 samples)."""
    distance = 0
    for i in range(187):
        distance += (v1[i] - v2[i])**2
    return math.sqrt(distance)

def predict(train_data, test_sample):
    """Finds the nearest neighbor's label."""
    min_dist = float('inf')
    label = -1
    for row in train_data:
        dist = euclidean_distance(row, test_sample)
        if dist < min_dist:
            min_dist = dist
            label = int(row[187]) # The 188th column is the class
    return label

# --- MAIN EXECUTION ---

path = "C:/Users/Hello/Desktop/model training/"

train_set = load_data(path + 'mitbih_train.csv', limit=2000) 
test_set = load_data(path + 'mitbih_test.csv', limit=50)

correct = 0
for sample in test_set:
    actual = int(sample[187])
    predicted = predict(train_set, sample)
    if actual == predicted:
        correct += 1

accuracy = (correct / len(test_set)) * 100
print(f"Accuracy: {accuracy}%")