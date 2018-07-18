import random
from PIL import Image
import numpy as np

from mdai import visualize


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


from keras.utils import Sequence, to_categorical
from skimage.transform import resize


class DataGenerator(Sequence):
    def __init__(
        self, dataset, batch_size=32, dim=(32, 32), n_channels=1, n_classes=10, shuffle=True
    ):
        """Generates data for Keras fit_generator() function.
        """

        # Initialization
        self.dim = dim
        self.batch_size = batch_size

        self.img_ids = dataset.image_ids
        self.imgs_anns = dataset.imgs_anns
        self.dataset = dataset

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.img_ids) / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"

        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        img_ids_temp = [self.img_ids[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(img_ids_temp)

        return X, y

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.img_ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, img_ids_temp):
        "Generates data containing batch_size samples"

        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(img_ids_temp):
            image = visualize.load_dicom_image(ID, to_RGB=True)
            image = resize(image, (self.dim[0], self.dim[1]))
            X[i,] = image

            ann = self.imgs_anns[ID][0]
            y[i] = self.dataset.labels_dict[ann["labelId"]]["class_id"]
        return X, to_categorical(y, num_classes=self.n_classes)
