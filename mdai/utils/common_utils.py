import random
import copy


def hex2rgb(h):
    """Convert Hex color encoding to RGB color"""
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def train_test_split(dataset, shuffle=True, validation_split=0.1):
    """
    Split image ids into training and validation sets.
    TODO: Need to update dataset.dataset_data!
    TODO: What to set for images_dir for combined dataset?
    """
    if validation_split < 0.0 or validation_split > 1.0:
        raise ValueError("{} is not a valid split ratio.".format(validation_split))

    image_ids_list = dataset.get_image_ids()
    if shuffle:
        sorted(image_ids_list)
        random.seed(42)
        random.shuffle(image_ids_list)

    split_index = int((1 - validation_split) * len(image_ids_list))
    train_image_ids = image_ids_list[:split_index]
    valid_image_ids = image_ids_list[split_index:]

    def filter_by_ids(ids, imgs_anns_dict):
        return {x: imgs_anns_dict[x] for x in ids}

    train_dataset = copy.deepcopy(dataset)
    train_dataset.id = dataset.id + "-TRAIN"

    valid_dataset = copy.deepcopy(dataset)
    valid_dataset.id = dataset.id + "-VALID"

    imgs_anns_dict = dataset.imgs_anns_dict

    train_imgs_anns_dict = filter_by_ids(train_image_ids, imgs_anns_dict)
    valid_imgs_anns_dict = filter_by_ids(valid_image_ids, imgs_anns_dict)

    train_dataset.image_ids = train_image_ids
    valid_dataset.image_ids = valid_image_ids

    train_dataset.imgs_anns_dict = train_imgs_anns_dict
    valid_dataset.imgs_anns_dict = valid_imgs_anns_dict

    print(
        "Num of instances for training set: %d, validation set: %d"
        % (len(train_image_ids), len(valid_image_ids))
    )
    return train_dataset, valid_dataset
