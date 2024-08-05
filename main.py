from mnist_reader import read_images, read_labels
from knn_classifier import predict
import config

def evaluate_accuracy(train_images, train_labels, test_images, test_labels, k):
    correct = 0
    total = len(test_images)
    predictions = []
    for i, test_image in enumerate(test_images):
        if i % config.PRINT_INTERVAL == 0:
            print(f"Processing image {i}/{total}")
        predicted_label = predict(train_images, train_labels, test_image, k)
        predictions.append(predicted_label)
        if predicted_label == test_labels[i]:
            correct += 1
    accuracy = correct / total
    return accuracy, predictions

if __name__ == "__main__":
    print("Loading training data...")
    train_images = read_images(config.TRAIN_IMAGES_PATH)
    train_labels = read_labels(config.TRAIN_LABELS_PATH)

    print("Loading test data...")
    test_images = read_images(config.TEST_IMAGES_PATH)
    test_labels = read_labels(config.TEST_LABELS_PATH)

    print(f"Evaluating model with k={config.K}, {config.N_TRAIN} training images, and {config.N_TEST} test images...")
    accuracy, predictions = evaluate_accuracy(train_images[:config.N_TRAIN], train_labels[:config.N_TRAIN],
                                              test_images[:config.N_TEST], test_labels[:config.N_TEST], config.K)

    print(f"Accuracy: {accuracy:.2%}")

    print(f"\nPredictions for the first {config.NUM_PREDICTIONS_TO_SHOW} test images:")
    for i in range(min(config.NUM_PREDICTIONS_TO_SHOW, len(predictions))):
        print(f"Image {i+1}: Predicted {predictions[i]}, Actual {test_labels[i]}")
