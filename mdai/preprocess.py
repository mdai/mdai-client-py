import numpy as np
import re
import os
import warnings
import json
import pandas as pd
import pydicom
import collections
import glob


class Project:
    """Project consists of label groups, and datasets.

    Args:
        annotations_fp (str):
            File path to the exported JSON annotation file.
        images_dir (str):
            File path to the DICOM images directory.
    """

    def __init__(self, annotations_fp=None, images_dir=None):
        """

        """
        self.annotations_fp = None
        self.images_dir = None
        self.label_groups = []
        self.datasets = []

        if annotations_fp is not None and images_dir is not None:
            self.annotations_fp = annotations_fp
            self.images_dir = images_dir

            with open(self.annotations_fp, "r") as f:
                self.data = json.load(f)

            for dataset in self.data["datasets"]:
                self.datasets.append(Dataset(dataset, images_dir))

            for label_group in self.data["labelGroups"]:
                self.label_groups.append(LabelGroup(label_group))
        else:
            print("Error: Missing data or images file paths!")

    def get_label_groups(self):
        return self.label_groups

    def show_label_groups(self):
        for label_group in self.label_groups:
            print("Label Group, Id: %s, Name: %s" % (label_group.id, label_group.name))
            label_group.show_labels("\t")

    def get_label_group_by_name(self, label_group_name):
        for label_group in self.label_groups:
            if label_group.name == label_group_name:
                return label_group
        return None

    def get_label_group_by_id(self, label_group_id):
        for label_group in self.label_groups:
            if label_group.id == label_group_id:
                return label_group
        return None

    def get_datasets(self):
        """Get JSON representation of datasets"""
        return self.datasets

    def show_datasets(self):
        print("Datasets:")
        for dataset in self.datasets:
            print("Id: %s, Name: %s" % (dataset.id, dataset.name))
        print("")

    def get_dataset_by_name(self, dataset_name):
        for dataset in self.datasets:
            if dataset.name == dataset_name:
                return dataset

        raise ValueError("Dataset name {} does not exist.".format(dataset_name))

    def get_dataset_by_id(self, dataset_id):
        for dataset in self.datasets:
            if dataset.id == dataset_id:
                return dataset
        raise ValueError("Dataset id {} does not exist.".format(dataset_id))

    def set_labels_dict(self, labels_dict):

        self.classes_dict = self._create_classes_dict(labels_dict)

        for dataset in self.datasets:
            dataset.classes_dict = self.classes_dict

    def get_label_id_annotation_mode(self, label_id):
        "Return label id's annotation mode."
        for label_group in self.label_groups:
            labels_data = label_group.get_data()["labels"]
            for label in labels_data:
                if label["id"] == label_id:
                    return label["annotationMode"]
        raise ValueError("Label id {} does not exist.".format(label_id))

    def get_label_id_type(self, label_id):
        "Return label id's type."
        for label_group in self.label_groups:
            labels_data = label_group.get_data()["labels"]
            for label in labels_data:
                if label["id"] == label_id:
                    return label["type"]
        raise ValueError("Label id {} does not exist.".format(label_id))

    def get_label_id_scope(self, label_id):
        "Return label id's scope."
        for label_group in self.label_groups:
            labels_data = label_group.get_data()["labels"]
            for label in labels_data:
                if label["id"] == label_id:
                    return label["scope"]
        raise ValueError("Label id {} does not exist.".format(label_id))

    def _create_classes_dict(self, labels_dict):
        """Create a dict with label id as key, and a nested dict of class_id, and class_text as \
        values, e.g., {'L_v8n': {'class_id': 1, 'class_text': 'Lung Opacity'}}, where L_v8n is \
        the label id, with a class_id of 1 and class text of 'Lung Opacity'.

        Args:
            labels_dict:
                dictionary containing label ids, and (user defined) class ids

        Returns:
            classes dict
        """
        classes_dict = {}

        for label_id, class_id in labels_dict.items():
            for label_group in self.label_groups:
                labels_data = label_group.get_data()["labels"]
                for label in labels_data:
                    if label["id"] == label_id:
                        if class_id == 0 and label["type"] == "local":
                            raise Exception(
                                "{} is a local type, its class id cannot be 0.".format(label_id)
                            )
                        classes_dict[label_id] = {
                            "class_id": class_id,
                            "class_text": label["name"],
                            "class_annotation_mode": label["annotationMode"],
                            "scope": label["scope"],
                            "type": label["type"],
                        }

        if classes_dict.keys() != labels_dict.keys():
            in_labels = labels_dict.keys()
            out_labels = classes_dict.keys()
            diff = set(in_labels).symmetric_difference(out_labels)
            raise ValueError("Labels {} are not valid for this dataset.".format(diff))

        return classes_dict


class LabelGroup:
    """A label group contains multiple labels.
    Each label has properties such id, name, color, type, scope, annotation mode, rad lex tag ids.

    Label type:
        Global typed annotations apply to the whole instance (e.g., a CT image), while
        local typed annotations apply to a part of the image (e.g., ROI bounding box).
    Label scope:
        Scope can be of study, series, or instance.
    Label annotation mode:
        Annotation mode can be of bounding boxes, free form, polygon, etc.
    """

    def __init__(self, label_group_data):
        """
        Args:
            label_group (object: json) JSON data for label group
        """
        self.label_group_data = label_group_data
        self.name = self.label_group_data["name"]
        self.id = self.label_group_data["id"]

    def get_data(self):
        return self.label_group_data

    def get_labels(self):
        """Get label ids and names """
        return [(label["id"], label["name"]) for label in self.label_group_data["labels"]]

    def show_labels(self, print_offset=""):
        """Show labels info"""
        print("{}Labels:".format(print_offset))
        for label in self.label_group_data["labels"]:
            print("{}Id: {}, Name: {}".format(print_offset, label["id"], label["name"]))
        print("")


class Dataset:
    """A dataset consists of DICOM images and annotations.
    Args:
        dataset_data:
            Dataset json data.
        images_dir:
            DICOM images directory.
    """

    def __init__(self, dataset_data, images_dir):

        self.dataset_data = dataset_data
        self.images_dir = images_dir

        self.id = dataset_data["id"]
        self.name = dataset_data["name"]
        self.all_annotations = dataset_data["annotations"]

        self.image_ids = None
        self.classes_dict = None
        self.imgs_anns_dict = None

        # all image ids
        self.all_image_ids = glob.glob(os.path.join(self.images_dir, "**/*.dcm"), recursive=True)

    def prepare(self):
        if self.classes_dict is None:
            raise Exception("Use `Project.set_labels_dict()` to set labels.")

        label_ids = self.classes_dict.keys()

        # filter annotations by label ids
        ann_filtered = self.get_annotations(label_ids)

        self.imgs_anns_dict = self._associate_images_and_annotations(ann_filtered)

    def get_annotations(self, label_ids=None, verbose=False):
        """Returns annotations, filtered by label ids.

        Args:
            label_ids (optional):
                Filter returned annotations by matching label ids.

            verbose (optional:
                Print debug messages.
        """
        if label_ids is None:
            if verbose:
                print("Dataset contains %d annotations." % len(self.all_annotations))
            return self.all_annotations

        ann_filtered = [a for a in self.all_annotations if a["labelId"] in label_ids]

        if verbose:
            print(
                "Dataset contains {} annotations, filtered by label ids {}.".format(
                    len(ann_filtered), label_ids
                )
            )
        return ann_filtered

    def _generate_uid(self, ann):
        """Generate an unique image identifier based on the DICOM file structure.

        Args:
            ann (list):
                List of annotations.

        Returns:
            A unique image identifier based on the DICOM file structure.
        """

        uid = None

        if "StudyInstanceUID" and "SeriesInstanceUID" and "SOPInstanceUID" in ann:
            # SOPInstanceUID aka image level
            uid = os.path.join(
                self.images_dir,
                ann["StudyInstanceUID"],
                ann["SeriesInstanceUID"],
                ann["SOPInstanceUID"] + ".dcm",
            )
            return uid
        elif "StudyInstanceUID" and "SeriesInstanceUID" in ann:
            prefix = os.path.join(
                self.images_dir, ann["StudyInstanceUID"], ann["SeriesInstanceUID"]
            )
            uid = [image_id for image_id in self.all_image_ids if image_id.startswith(prefix)]
            # print("SeriesInstanceUID {}, uid {}".format(ann["SeriesInstanceUID"], uid))
            return uid
        elif "StudyInstanceUID" in ann:
            prefix = os.path.join(self.images_dir, ann["StudyInstanceUID"])
            uid = [image_id for image_id in self.all_image_ids if image_id.startswith(prefix)]
            # print("StudyInstanceUID {}, uid {}".format(ann["StudyInstanceUID"], uid))
            return uid
        else:
            raise ValueError("Unable to create UID from {}".format(ann))

    def get_image_ids(self, verbose=False):
        """Returns image ids. Must call prepare() method first in order to generate image ids.

        Args:
            verbose (Optional):
                Print debug message.
        """
        if not self.image_ids:
            raise Exception("Call project.prepare() first.")

        if verbose:
            print(
                "Dataset contains {} images, filtered by label ids {}.".format(
                    len(self.image_ids), self.classes_dict.keys()
                )
            )
        return self.image_ids

    def _generate_image_ids(self, anns):
        """Get images ids for annotations.

        Args:
            ann (list):
            List of image ids.

        Returns:
            A list of image ids.
        """
        image_ids = set()
        for ann in anns:
            uid = self._generate_uid(ann)

            if uid:
                if isinstance(uid, list):
                    for one_uid in uid:
                        image_ids.add(one_uid)
                else:
                    image_ids.add(uid)

        # image_ids = glob.glob(os.path.join(self.images_dir, "**/*.dcm"), recursive=True)
        return sorted(list(image_ids))

    def get_annotations_by_image_id(self, image_id):
        if image_id not in self.image_ids:
            raise ValueError("Image id {} is not found in dataset {}.".format(image_id, self.name))

        return self.imgs_anns_dict[image_id]

    def _associate_images_and_annotations(self, anns):
        """Generate image ids to annotations mapping.
        Each image can have zero or more annotations.

        Args:
            anns (list):
                List of annotations.

        Returns:
            A dictionary with image ids as keys and annotations as values.
        """
        self.image_ids = self._generate_image_ids(anns)

        # empty dictionary with image ids as keys
        imgs_anns_dict = collections.OrderedDict()
        imgs_anns_dict = {fp: [] for fp in self.image_ids}

        for ann in anns:
            uid = self._generate_uid(ann)
            if uid:
                if isinstance(uid, list):
                    for one_uid in uid:
                        imgs_anns_dict[one_uid].append(ann)
                else:
                    imgs_anns_dict[uid].append(ann)

        return imgs_anns_dict

    def class_id_to_class_text(self, class_id):
        for k, v in self.classes_dict.items():
            if v["class_id"] == class_id:
                return v["class_text"]

        raise Exception("class_id {} is invalid.".format(class_id))

    def class_text_to_class_id(self, class_text):
        for k, v in self.classes_dict.items():
            if v["class_text"] == class_text:
                return v["class_id"]
        raise Exception("class_text {} is invalid.".format(class_text))

    def label_id_to_class_id(self, label_id):
        for k, v in self.classes_dict.items():
            if k == label_id:
                return v["class_id"]
        raise Exception("label_id {} is invalid.".format(label_id))

    def label_id_to_class_annotation_mode(self, label_id):
        for k, v in self.classes_dict.items():
            if k == label_id:
                return v["class_annotation_mode"]
        raise Exception("label_id {} is invalid.".format(label_id))

    def show_classes(self):
        for k, v in self.classes_dict.items():
            print(
                "Label id: {}, Class id: {}, Class text: {}".format(
                    k, v["class_id"], v["class_text"]
                )
            )
