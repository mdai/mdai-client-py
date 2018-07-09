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

class Dataset(object): 
    """
    A dataset consists of annotations, labels and associated DICOM images. 
    
    A dataset can contain multiple multiple label groups. User should select which 
    label group to use. To show all label groups, call show_label_groups() function 
    on a Dataset object instance. To set a label group, call set_label_group() function 
    with desired label group id. 

    Local vs. Global Annotation Type: 
    Global annotations apply to the whole instance (e.g., a CT image), while
    local annottations apply to a part of the image (e.g., bounding box around something 
    in the image.)

    """ 

    label_group_id = -1 
    project_data_fp = None 
    project_images_fp = None 

    def __init__(self, project_data_fp=None, project_images_fp=None): 
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

    def get_annotations(self):
        print("Dataset contains %d annotations." % len(self.annotations))
        return self.annotations

    def show_labels(self): 
        """Show labels for selected label group."""
        if self.label_group_id == -1: 
            print('Need to specify Label Group Id with set_label_group()')
            self.show_label_groups() 
        print('Labels:')
        for label in self.labels: 
            print("id: %s, name: %s, type: %s" % (label['id'], label['name'], label['type']))

    def get_filtered_annotations(self, label_ids=None):
        """Get filtered annotations for specified label_ids."""
        if label_ids is None: 
            print('Error: Need to specify label_ids!')
            return None 

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
 
    def _generate_id(self, a):
        """Generate an unique image identifier
        """ 
        uid = None
        try: 
            uid = os.path.join(self.project_images_fp, a['StudyInstanceUID'], a['SeriesInstanceUID'], a['SOPInstanceUID']) + '.dcm'
        except Exception as error: 
            print('Exception:', error)
            print('a %s'% a)
        return uid  

    def get_image_ids(self, annotations_filtered):
        """Get images ids for filtered annotations."""
        image_ids = set()
        for a in annotations_filtered:
            uid = self._generate_id(a) 
            if uid: 
                image_ids.add(uid)
        return list(image_ids)

    def associate_images_and_annotations(self, image_fps, local_label_ids, annotations_filtered): 
        """Associate image annotations with the corresponding image,
           using a unique file path identifier. 
        """
        local_image_annotations = {fp: [] for fp in image_fps}
        for a in annotations_filtered:
            # only add local annotations with data
            if a['labelId'] in local_label_ids and a['data'] is not None:
                uid = self._generate_id(a)
                local_image_annotations[uid].append(a)
        return local_image_annotations
