import numpy as np

class ART1:
    def __init__(self, num_features, num_clusters, vigilance=0.5):
        self.num_features = num_features
        self.num_clusters = num_clusters
        self.vigilance = vigilance
        self.weights = np.ones((num_clusters, num_features * 2))  # Using complement coding

    def complement_coding(self, input_pattern):
        return np.concatenate((input_pattern, 1 - input_pattern))

    def calculate_similarity(self, input_pattern, cluster_weights):
        return np.sum(np.minimum(input_pattern, cluster_weights)) / np.sum(input_pattern)

    def train(self, data):
        data = np.array([self.complement_coding(pattern) for pattern in data])
        for input_pattern in data:
            while True:
                similarities = [self.calculate_similarity(input_pattern, w) for w in self.weights]
                selected_cluster = np.argmax(similarities)

                if similarities[selected_cluster] >= self.vigilance:
                    self.weights[selected_cluster] = np.minimum(self.weights[selected_cluster], input_pattern)
                    break
                else:
                    self.weights[selected_cluster] = np.zeros_like(self.weights[selected_cluster])

    def predict(self, data):
        data = np.array([self.complement_coding(pattern) for pattern in data])
        predictions = []
        for input_pattern in data:
            similarities = [self.calculate_similarity(input_pattern, w) for w in self.weights]
            predictions.append(np.argmax(similarities))
        return predictions

# Example Binary Data
data = np.array([
    [1, 0, 0, 1],
    [1, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 1]
])

# Initialize and Train ART1 Network
art = ART1(num_features=4, num_clusters=4, vigilance=0.7)
art.train(data)

# Predict Cluster Assignments
predictions = art.predict(data)
print("Predicted Clusters:", predictions)

print("\nFinal Weights:")
print(art.weights)
