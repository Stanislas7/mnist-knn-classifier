def read_images(filename, num_images_to_read=None):
    with open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')  # Magic number (should be 2051 for images)
        total_images = int.from_bytes(f.read(4), 'big')  # Total number of images
        num_rows = int.from_bytes(f.read(4), 'big')  # Number of rows per image
        num_cols = int.from_bytes(f.read(4), 'big')  # Number of columns per image

        # Determine how many images to read
        num_images = num_images_to_read if num_images_to_read else total_images

        # Read the image data
        images = []
        for _ in range(num_images):
            # Each image is 28x28 pixels = 784 bytes
            image = []
            for _ in range(num_rows * num_cols):
                pixel = int.from_bytes(f.read(1), 'big')  # Read one byte (one pixel)
                image.append(pixel)
            images.append(image)

    return images

def read_labels(filename, num_labels_to_read=None):
    with open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')  # Magic number (should be 2049 for labels)
        total_labels = int.from_bytes(f.read(4), 'big')  # Total number of labels

        # Determine how many labels to read
        num_labels = num_labels_to_read if num_labels_to_read else total_labels

        # Read the label data
        labels = []
        for _ in range(num_labels):
            label = int.from_bytes(f.read(1), 'big')  # Read one byte (one label)
            labels.append(label)

    return labels

def read_test_images(filename, num_images_to_read=None):
    return read_images(filename, num_images_to_read)

def read_test_labels(filename, num_labels_to_read=None):
    return read_labels(filename, num_labels_to_read)
