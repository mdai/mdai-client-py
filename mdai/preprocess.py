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

def get_label_ids(labels, local_only=False):
    """ Get label ids for labels. """
    if local_only: 
        return [label['id']for label in labels if label['type'] == 'local' ]
    else: 
        return [label['id'] for label in labels]

# TODO: rename 
def get_label_ids_dict(label_ids, labels): 
    """Create a dict with label id as key, and 
       a nested dict of class_id, and class_text as values 
       e.g., {'L_v8n': {'class_id': 1, 'class_text': 'Lung Opacity'}}
       where L_v8n is the label id, with a class_id of 1 and class text of 'Lung Opacity'
    """ 
    label_ids_dict = {}
    for i, llid in enumerate(label_ids):
        for label in labels: 
            if label['id'] == llid: 
                class_text = label['name']
                
        label_ids_dict[llid] = {'class_id': i+1, 'class_text': class_text}

    return label_ids_dict

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

    label_group_id = -1 
    project_data_fp = None 
    project_images_fp = None 

    def __init__(self, project_data_fp=None, project_images_fp=None): 
        """ project_data_fp is the annotation, 
            project_images_fp is the DICOM image directory. 
        """
        if project_data_fp is not None and project_images_fp is not None: 
            self.project_data_fp = project_data_fp
            self.project_images_fp = project_images_fp

            with open(project_data_fp, 'r') as f:
                self.data = json.load(f)
        self.annotations = self.data['datasets'][0]['annotations']

    def _is_valid_label_group(self, label_group_id):
        """Check to see if label group id is valid. """
        num_label_groups = len(self.data['labelGroups'])
        if label_group_id >= 0 and label_group_id < num_label_groups:  
            return True
        else: 
            return False

    def set_label_group(self, label_group_id):
        """Set desired label group. """ 
        if self._is_valid_label_group(label_group_id): 
            print('Setting label group to %s, Id: %d' % (self.data['labelGroups'][label_group_id]['name'], label_group_id))
            self.label_group_id = label_group_id
            return True 
        else: 
            print('Label Group id %d is invalid' % (label_group_id))
            return False 

    def show_label_groups(self): 
        """Show avaiable label groups.""" 
        print('Available Label Groups are:')
        for i, label_group in enumerate(self.data['labelGroups']):
            print('Id: %d, Name: %s' % (i, label_group['name'])) 

    def get_labels(self, label_group_id=-1): 
        """Once label group is selected, get labels for the label group."""
        if label_group_id == -1 and self.label_group_id != -1: 
            print('Getting labels for Label Group id: %d' % self.label_group_id)
            self.labels = self.data['labelGroups'][self.label_group_id]['labels']
            return self.labels

        # if no label_group_id is specified 
        if label_group_id == -1: 
            print('Please specify label group id using set_label_group():')
            self.show_label_groups() 
            return None 

        else: 
            if self._is_valid_label_group(label_group_id): 
                print('Getting labels for Label Group id: ', label_group_id)
                self.labels = self.data['labelGroups'][label_group_id]['labels']
                return self.labels
            else: 
                print('Invalid label group id')
                self.show_label_groups() 
                return None 

    def show_labels(self): 
        """Show labels for selected label group."""
        if self.label_group_id == -1: 
            print('Need to specify Label Group Id with set_label_group()')
            self.show_label_groups() 
        print('Labels:')
        for label in self.labels: 
            print("id: %s, name: %s, type: %s" % (label['id'], label['name'], label['type']))


    def get_annotations(self, label_ids=None):
        """
            Filtered annotations for specified label_ids.
        """
        #print("Dataset contains %d annotations." % len(self.annotations))

        if label_ids is None: 
            return self.annotations

        if self.label_group_id == -1: 
            print('Need to specify Label Group Id with set_label_group()')
            self.show_label_groups() 
            return None

        self.labels = self.data['labelGroups'][self.label_group_id]['labels']

        print('Filtered annotations for these labels:')
        for label in self.labels: 
            if label['id'] in label_ids: 
                print(label['name'])
        print('\n')
        
        annotations_filtered = [a for a in self.annotations if a['labelId'] in label_ids]
        print("Filtered Dataset contains %d annotations." % len(annotations_filtered))
        return annotations_filtered
 
    def _generate_uid(self, ann):
        """Generate an unique image identifier
        """ 
        uid = None
        try: 
            uid = os.path.join(self.project_images_fp, ann['StudyInstanceUID'], 
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

    def associate_images_and_annotations(self, image_fps, local_label_ids, ann): 
        """Associate image annotations with the corresponding image,
           using a unique file path identifier. 
        """
        images_and_annotations = {fp: [] for fp in image_fps}
        for a in ann:
            # only add local annotations with data
            if a['labelId'] in local_label_ids and a['data'] is not None:
                uid = self._generate_uid(a)
                images_and_annotations[uid].append(a)
        return images_and_annotations
