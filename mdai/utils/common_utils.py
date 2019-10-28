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
        raise ValueError(f"{validation_split} is not a valid split ratio.")

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


def json_to_dataframe(json_file, datasets=[]):
    with open(json_file, "r") as f:
        data = json.load(f)

    a = pd.DataFrame([])
    studies = pd.DataFrame([])
    labels = None

    # Gets annotations for all datasets
    for d in data["datasets"]:
        if d["name"] in datasets or len(datasets) == 0:
            study = pd.DataFrame(d["studies"])
            study["dataset"] = d["name"]
            studies = studies.append(study, ignore_index=True, sort=False)

            annots = pd.DataFrame(d["annotations"])
            annots["dataset"] = d["name"]
            a = a.append(annots, ignore_index=True, sort=False)

    if len(studies) > 0:
        studies = studies[["StudyInstanceUID", "dataset", "number"]]
    g = pd.DataFrame(data["labelGroups"])
    if len(g) > 0:
        # unpack arrays
        result = pd.DataFrame([(d, tup.id, tup.name) for tup in g.itertuples() for d in tup.labels])
        result.columns = ["labels", "groupId", "groupName"]

        def unpack_dictionary(df, column):
            ret = None
            ret = pd.concat(
                [df, pd.DataFrame((d for idx, d in df[column].iteritems()))], axis=1, sort=False
            )
            del ret[column]
            return ret

        labels = unpack_dictionary(result, "labels")
        labels = labels[[
                "groupId",
                "groupName",
                "annotationMode",
                "color",
                "description",
                "id",
                "name",
                "radlexTagIds",
                "scope",
            ]]
        labels.columns = [
            "groupId",
            "groupName",
            "annotationMode",
            "color",
            "description",
            "labelId",
            "labelName",
            "radlexTagIdsLabel",
            "scope"
        ]

        a = a.merge(labels, on="labelId", sort=False)
    a = a.merge(studies[["StudyInstanceUID", "number"]], on="StudyInstanceUID", sort=False)
    return {'annotations': a, 'studies': studies, 'labels': labels}
