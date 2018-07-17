import numpy as np 
import re 
import os 
import warnings 
import json 

import pandas as pd

import pydicom 

import logging
logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)

# def get_label_ids(labels, local_only=False):
#     """ Get label ids for labels. """
#     if local_only: 
#         return [label['id']for label in labels if label['type'] == 'local' ]
#     else: 
#         return [label['id'] for label in labels]

# # TODO: rename 
# def get_label_ids_dict(label_ids, labels): 
#     """Create a dict with label id as key, and 
#        a nested dict of class_id, and class_text as values 
#        e.g., {'L_v8n': {'class_id': 1, 'class_text': 'Lung Opacity'}}
#        where L_v8n is the label id, with a class_id of 1 and class text of 'Lung Opacity'
#     """ 
#     label_ids_dict = {}
#     for i, llid in enumerate(label_ids):
#         for label in labels: 
#             if label['id'] == llid: 
#                 class_text = label['name']
                
#         label_ids_dict[llid] = {'class_id': i+1, 'class_text': class_text}

#     return label_ids_dict

class Project(object):

    def __init__(self, data_fp=None, images_fp=None):
        self.data_fp = None 
        self.images_fp = None 
        self.label_groups = [] 
        self.datasets = []

        if data_fp is not None and images_fp is not None: 
            self.data_fp = data_fp
            self.images_fp = images_fp

        with open(self.data_fp, 'r') as f:
                self.data = json.load(f)
        
        for dataset in self.data['datasets']:
            self.datasets.append(Dataset(dataset, images_fp))

        for label_group in self.data['labelGroups']:
            self.label_groups.append(LabelGroup(label_group))

    def get_label_groups(self):
        #print('Available Label Groups are:')
        #label_groups_df = pd.DataFrame(self.data['labelGroups'])
        #return label_groups_df

        # for label_group in self.label_groups:
        #      print('Name: %s, Id: %s' % (label_group.name, label_group.id)) 
        #      print(label_group.get_labels())
        return self.label_groups

    def get_label_group_by_id(self, label_group_id): 
        for label_group in self.label_groups: 
            if label_group.id == label_group_id: 
                return label_group
        return None
    
    def get_datasets(self):
        return self.datasets

    def get_dataset_by_name(self, dataset_name):
        for dataset in self.datasets: 
            if dataset.name == dataset_name: 
                return dataset
        return None 

class LabelGroup(): 
    def __init__(self, label_group):
        self.label_group_data = label_group
        self.name = self.label_group_data['name']
        self.id = self.label_group_data['id']

    def get_labels(self): 
        return [(label['id'], label['name']) for label in self.label_group_data['labels']]

class Dataset(object): 
    """
    A dataset consists of DICOM images and annotations. 
    
    A dataset can contain multiple multiple label groups. User should select which 
    label group to use. To show all label groups, call show_label_groups() function 
    on a Dataset object instance. To set a label group, call set_label_group() function 
    with desired label group id. 

    Local vs. global annotation type for a label: 
    For a label, lobal annotations apply to the whole instance (e.g., a CT image), while
    local annottations apply to a part of the image (e.g., ROI bounding box).
    """ 

    def __init__(self, dataset_data, images_fp): 
        """ images_fp is the DICOM image directory. 
        """
        self.dataset_data = dataset_data 
        self.images_fp = images_fp

        self.id = dataset_data['id']
        self.name = dataset_data['name']
        self.annotations = dataset_data['annotations']
   
    def get_annotations(self, label_ids=None):
        if label_ids is None: 
            return self.annotations
        
        annotations_filtered = [a for a in self.annotations if a['labelId'] in label_ids]
        print("Filtered Dataset contains %d annotations." % len(annotations_filtered))
        return annotations_filtered
 
    def _generate_uid(self, ann):
        """Generate an unique image identifier
        """ 
        uid = None
        try: 
            uid = os.path.join(self.images_fp, ann['StudyInstanceUID'], 
                               ann['SeriesInstanceUID'], ann['SOPInstanceUID']) + '.dcm'
        except Exception as error: 
            print('Exception:', error)
            print('ann %s'% ann)
        return uid  

    def get_image_ids(self, ann):
        """Get images ids for annotations."""
        image_ids = set()
        for a in ann:
            uid = self._generate_uid(a) 
            if uid: 
                image_ids.add(uid)
        return list(image_ids)

    def associate_images_and_annotations(self, ann, label_ids): 
        """Associate image annotations with the corresponding image,
           using a unique file path identifier. 
        """
        image_ids = self.get_image_ids(ann)
        images_and_annotations = {fp: [] for fp in image_ids}
        for a in ann:
            # only add local annotations with data
            if a['labelId'] in label_ids:
                uid = self._generate_uid(a)
                images_and_annotations[uid].append(a)
        return images_and_annotations
