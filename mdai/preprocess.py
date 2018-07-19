import numpy as np
import re
import os
import warnings
import json
import pandas as pd
import pydicom


class Project:
    """Project consists of label groups, and datasets.
    """

    def __init__(self, annotations_fp=None, images_dir=None):
        """ 
        Args: 
            annotations_fp (str): File path to the exported JSON annotation file. 
            images_dir (str): File path to the DICOM images directory. 
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

    def show_label_groups_info(self):
        print("Label Groups:")
        for label_group in self.label_groups:
            print("Name: %s, Id: %s" % (label_group.name, label_group.id))
            print("\t")
            label_group.show_labels_info()
        print("")

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
        return self.datasets

    def show_datasets_info(self):
        print("Datasets:")
        for dataset in self.datasets:
            print("Name: %s, Id: %s" % (dataset.name, dataset.id))
        print("")

    def get_dataset_by_name(self, dataset_name):
        for dataset in self.datasets:
            if dataset.name == dataset_name:
                return dataset
        return None

    def get_dataset_by_id(self, dataset_id):
        for dataset in self.datasets:
            if dataset.id == dataset_id:
                return dataset
        return None

    def _check_label_ids(self, label_ids):
        pass

    def set_label_ids(self, selected_label_ids):
        self.selected_label_ids = selected_label_ids
        self.labels_dict = self._create_labels_dict()

        for dataset in self.datasets:
            dataset.selected_label_ids = selected_label_ids
            dataset.labels_dict = self.labels_dict

    def _create_labels_dict(self):
        """Create a dict with label id as key, and a nested dict of class_id, and class_text 
        as values, e.g., {'L_v8n': {'class_id': 1, 'class_text': 'Lung Opacity'}}, 
        where L_v8n is the label id, with a class_id of 1 and class text of 'Lung Opacity'.

        Args: 
            label_ids (list): list of label ids. 
            labels (list): list of tuples of (label_id, label_name)
        
        Returns: 
            Label ids dictionary.
        """
        label_ids = self.selected_label_ids

        label_ids_dict = {}
        for i, label_id in enumerate(label_ids):

            # find which label group has the label
            for label_group in self.label_groups:
                labels = label_group.get_labels()
                for label in labels:
                    if label[0] == label_id:
                        class_text = label[1]
            label_ids_dict[label_id] = {"class_id": i, "class_text": class_text}
        return label_ids_dict


class LabelGroup:
    """A label group contains multiple labels.  
    
    Each label has properties such id, name, color, type, scope, annotation mode, rad lex tag ids.
    
    Label type: Global typed annotations apply to the whole instance (e.g., a CT image),
    while local typed annotations apply to a part of the image (e.g., ROI bounding box).

    Label scope: Scope can be of study, series, or instance.

    Label annotation mode: can be of bounding boxes, free form , polgon, etc.
    """

    def __init__(self, label_group):
        """
        Args: 
            label_group (object: json) JSON data for label group
        """
        self.label_group_data = label_group
        self.name = self.label_group_data["name"]
        self.id = self.label_group_data["id"]

    def get_labels(self):
        """Get label ids and names """
        return [(label["id"], label["name"]) for label in self.label_group_data["labels"]]

    def show_labels_info(self):
        """Show labels info"""
        print("Labels:")
        for label in self.label_group_data["labels"]:
            print("Name: %s, Id: %s" % (label["name"], label["id"]))
        print("")

    def get_label_group_dict(self):
        """Raw label group JSON data"""
        return self.label_group_data


class Dataset:
    """
    A dataset consists of DICOM images and annotations.
    """

    def __init__(self, dataset_data, images_dir):
        """ 
        Args: 
            dataset_data: Dataset json data
            images_dir: DICOM images directory. 
        """
        self.dataset_data = dataset_data
        self.images_dir = images_dir

        self.id = dataset_data["id"]
        self.name = dataset_data["name"]
        self.annotations = dataset_data["annotations"]
        self.selected_label_ids = None
        self.labels_dict = None

    def prepare(self):
        if self.selected_label_ids is None:
            raise Exception(
                "Label ids must be selected. Use select_label_ids() to \
                             set project wide label ids."
            )

        ann = self.get_annotations(self.selected_label_ids)
        self.imgs_anns = self._associate_images_and_annotations(ann)

    def get_annotations(self, label_ids):
        """Returns annotations, filtered by label ids.
        """
        if label_ids is None:
            print("Dataset contains %d annotations." % len(self.annotations))
            return self.annotations

        # TODO: print label ids
        ann_filtered = [a for a in self.annotations if a["labelId"] in label_ids]
        print("Dataset contains %d annotations." % len(ann_filtered))
        return ann_filtered

    def _generate_uid(self, ann):
        """Generate an unique image identifier based on the file path and DICOM structure.

        Args:
            ann (list): List of annotations.
        
        Returns:
            A unique image id.
        """
        uid = None
        try:
            uid = (
                os.path.join(
                    self.images_dir,
                    ann["StudyInstanceUID"],
                    ann["SeriesInstanceUID"],
                    ann["SOPInstanceUID"],
                )
                + ".dcm"
            )
        except Exception as error:
            print("Exception:", error)
            print("ann %s" % ann)
        return uid

    def get_image_ids(self):
        return self.image_ids

    def _get_image_ids(self, ann):
        """Get images ids for annotations.

        Args: 
            ann (list): List of annotations.

        Returns: 
            A list of image ids.
        """
        image_ids = set()
        for a in ann:
            uid = self._generate_uid(a)
            if uid:
                image_ids.add(uid)
        return list(image_ids)

    def _associate_images_and_annotations(self, ann):
        """Build a dictionary with image ids and annotations, filtered by label ids.

        Args: 
            ann (list): List of annotations.
            label_ids (list): List of label ids. Annotations is filtered based on label ids.
        
        Returns:
            Dictionary with image ids as keys and annotations as values.
        """
        self.image_ids = self._get_image_ids(ann)

        # empty dictionary with image ids as keys
        imgs_anns_dict = {fp: [] for fp in self.image_ids}
        for a in ann:
            uid = self._generate_uid(a)
            imgs_anns_dict[uid].append(a)
        return imgs_anns_dict

    def class_id_to_class_text(self, class_id):
        for k, v in self.labels_dict.items():
            if v["class_id"] == class_id:
                return v["class_text"]

