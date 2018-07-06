import numpy as np 
import re 
import os 
import warnings 
import json 

import pandas as pd

import pydicom 

import logging

#from logging.config import fileConfig
#log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logger.config')
#fileConfig(log_file_path)

_LOGGER = logging.getLogger(__name__)

handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
_LOGGER.addHandler(handler)
_LOGGER.setLevel(logging.DEBUG)

class Dataset(object): 
    label_group_id = -1 
    PROJECT_DATA_FP = None 
    PROJECT_IMAGES_FP = None 

    def __init__(self, PROJECT_DATA_FP=None, PROJECT_IMAGES_FP=None): 
        if PROJECT_DATA_FP is not None and PROJECT_IMAGES_FP is not None: 
            self.PROJECT_DATA_FP = PROJECT_DATA_FP
            self.PROJECT_IMAGES_FP = PROJECT_IMAGES_FP

            with open(PROJECT_DATA_FP, 'r') as f:
                self.data = json.load(f)
        self.annotations = self.data['datasets'][0]['annotations']

    def _is_valid_label_group(self, label_group_id):
        """Check to see if label group id is valid. 
        """
        num_label_groups = len(self.data['labelGroups'])
        if label_group_id >= 0 and label_group_id < num_label_groups:  
            return True
        else: 
            return False

    def set_label_group(self, label_group_id):
        if self._is_valid_label_group(label_group_id): 
            _LOGGER.info('Setting label group to %s, id %d' % (self.data['labelGroups'][label_group_id]['name'], label_group_id))
            self.label_group_id = label_group_id
            return True 
        else: 
            _LOGGER.error('Label Group id %d is invalid' % (label_group_id))
            return False 

    def show_label_groups(self): 
        print('Label Groups are:')
        for i, label_group in enumerate(self.data['labelGroups']):
            print('id: %d, name, %s' % (i, label_group['name'])) 

    def get_labels(self, label_group_id=-1): 

        if label_group_id == -1 and self.label_group_id != -1: 
            print('Getting labels for Label Group id: %d' % self.label_group_id)
            self.labels = self.data['labelGroups'][self.label_group_id]['labels']
            return self.labels

        # none specified 
        if label_group_id == -1: 
            print('Please specify label group id:')
            self.show_label_groups() 
            return None 

        else: 
            if self._is_valid_label_group(label_group_id): 
                print('Getting labels for Label Group id: ', label_group_id)
                self.labels = self.data['labelGroups'][label_group_id]['labels']
                return self.labels
            else: 
                _LOGGER.error('Invalid label group id')
                self.show_label_groups() 
                return None 

    def get_annotations(self):
        print("Dataset contains %d annotations." % len(self.annotations))
        return self.annotations

    def show_labels(self): 
        if self.label_group_id == -1: 
            print('Need to specify Label Group Id with set_label_group()')
            self.show_label_groups() 
        print('Labels:')
        for label in self.labels: 
            print("id: %s, name: %s, type: %s" % (label['id'], label['name'], label['type']))

    def get_filtered_annotations(self, label_ids=None):
        if label_ids is None: 
            print('Need to specify label_ids')
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
            uid = os.path.join(self.PROJECT_IMAGES_FP, a['StudyInstanceUID'], a['SeriesInstanceUID'], a['SOPInstanceUID']) + '.dcm'
        except Exception as error: 
            print('Exception:', error)
            print('a %s'% a)
        return uid  

    def get_image_ids(self, annotations_filtered):
        """Return 
        """
        image_ids = set()
        for a in annotations_filtered:
            uid = self._generate_id(a) 
            if uid: 
                image_ids.add(uid)
        return list(image_ids)

    # Todo: this should be renamed, it really just does association 
    # rename to combine
    def associate_images_and_annotations(self, image_fps, local_label_ids, annotations_filtered): 
        """Associate image annotations with the corresponding image,
           using a unique file path identifier
        """
        local_image_annotations = {fp: [] for fp in image_fps}
        for a in annotations_filtered:
            # only add local annotations with data
            if a['labelId'] in local_label_ids and a['data'] is not None:
                uid = self._generate_id(a)
                local_image_annotations[uid].append(a)
        return local_image_annotations