import random
import copy
import json
import pandas as pd


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


def json_to_dataframe(json_file, dataset="all_datasets", should_return_labels=False):
    with open(json_file, "r") as f:
        data = json.load(f)

    a = pd.DataFrame([])
    studies = pd.DataFrame([])

    # Gets annotations for all datasets
    if dataset == "all_datasets":
        for d in data["datasets"]:
            annots = pd.DataFrame(d["annotations"])
            annots["dataset"] = d["name"]
            study = pd.DataFrame(d["studies"])
            study["dataset"] = d["name"]
            a = a.append(annots, ignore_index=True, sort=False)
            studies = studies.append(study, ignore_index=True, sort=False)
    else:
        for d in data["datasets"]:
            if d["name"] == dataset:
                annots = pd.DataFrame(d["annotations"])
                annots["dataset"] = d["name"]
                study = pd.DataFrame(d["studies"])
                study["dataset"] = d["name"]
                a = a.append(annots, ignore_index=True, sort=False)
                studies = studies.append(study, ignore_index=True, sort=False)

    studies = studies[["StudyInstanceUID", "dataset", "number"]]
    g = pd.DataFrame(data["labelGroups"])

    # unpack arrays
    result = pd.DataFrame(
        [(d, tup.id, tup.name) for tup in g.itertuples() for d in tup.labels]
    )
    result.columns = ["labels", "id", "name"]

    def unpack_dictionary(df, column):
        ret = None
        ret = pd.concat(
            [df, pd.DataFrame((d for idx, d in df[column].iteritems()))],
            axis=1,
            sort=False,
        )
        del ret[column]
        return ret

    label_groups = unpack_dictionary(result, "labels")
    label_groups.columns = [
        "groupId",
        "groupName",
        "annotationMode",
        "color",
        "createdAt",
        "description",
        "labelId",
        "labelName",
        "radlexTagIdsLabel",
        "scope",
        "type",
        "updatedAt",
    ]
    label_groups = label_groups[
        [
            "groupId",
            "groupName",
            "annotationMode",
            "color",
            "description",
            "labelId",
            "labelName",
            "radlexTagIdsLabel",
            "scope",
        ]
    ]

    a = a.merge(label_groups, on="labelId", sort=False)
    a = a.merge(
        studies[["StudyInstanceUID", "number"]], on="StudyInstanceUID", sort=False
    )
    if should_return_labels == True:
        return a, studies, label_groups
    else:
        return a, studies
