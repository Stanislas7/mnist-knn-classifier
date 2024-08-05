import math

def euclidean_distance(img1, img2):
    """Return the Euclidean distance between two images."""
    return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(img1, img2)))

def get_k_nearest(train_images, train_labels, test_image, k):
    """Return the k nearest neighbors of the test image."""
    distances = []

    for i in range(len(train_images)):
        train_image = train_images[i]
        label = train_labels[i]

        distance = euclidean_distance(train_image, test_image)

        distances.append((distance, label))

    # Sort the distances list based on the distance
    sorted_distances = sorted(distances, key=lambda x: x[0])

    return sorted_distances[:k]

def weighted_vote(neighbors):
    """Return the label with the highest weighted vote."""
    vote_weights = {}
    for neighbor in neighbors:
        distance = neighbor[0]
        label = neighbor[1]

        # Calculate weight (inverse of distance)
        weight = 1.0 / (distance + 0.00001)  # Adding a small value to avoid division by zero

        # Adding weight to the label's total
        if label in vote_weights:
            vote_weights[label] += weight
        else:
            vote_weights[label] = weight

    # Find the label with the highest weight -> that's our prediction
    max_weight = 0
    max_label = None
    for label, weight in vote_weights.items():
        if weight > max_weight:
            max_weight = weight
            max_label = label

    return max_label

def predict(train_images, train_labels, test_image, k):
    """Return the predicted label for the test image."""
    neighbors = get_k_nearest(train_images, train_labels, test_image, k)
    return weighted_vote(neighbors)
