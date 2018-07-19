import random


def hex2rgb(h):
    """Convert Hex color encoding to RGB color"""
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def train_test_split(image_ids, shuffle=True, validation_split=0.1):
    """Split image ids into training and validation sets."""
    if validation_split < 0.0 or validation_split > 1.0:
        return None, None

    image_ids_list = list(image_ids)
    if shuffle:
        sorted(image_ids_list)
        random.seed(42)
        random.shuffle(image_ids_list)

    split_index = int((1 - validation_split) * len(image_ids_list))
    image_ids_train = image_ids_list[:split_index]
    image_ids_val = image_ids_list[split_index:]

    print(
        "Num of instances for training set: %d, validation set: %d"
        % (len(image_ids_train), len(image_ids_val))
    )
    return image_ids_train, image_ids_val
