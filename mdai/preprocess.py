import numpy as np 
import re 
import os 
import warnings 
import json 

import pandas as pd 

import pydicom 

class Project(object): 
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
        all_label_groups = len(self.data['labelGroups'])
        if label_group_id >= 0 and label_group_id < all_label_groups:  
            return True
        else: 
            return False

    def set_label_group(self, label_group_id):
        if self._is_valid_label_group(label_group_id): 
            print('Setting label group to %s, id %d' % (self.data['labelGroups'][label_group_id]['name'], label_group_id))
            self.label_group_id = label_group_id
            return True 
        else: 
            print('Label Group id %d is invalid' % (label_group_id))
            return False 

    def show_label_groups(self): 
        print('Label Groups are:')
        for i, label_group in enumerate(self.data['labelGroups']):
            print('id: %d, name, %s' % (i, label_group['name'])) 

    def get_labels(self, label_group_id=-1): 

        if label_group_id == -1 and self.label_group_id != -1: 
            print('Getting labels for Label Group id: ', self.label_group_id)
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
                print('Invalid label group id')
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

    def get_image_filepaths(self, annotations_filtered): 
        image_fps = set()

        for a in annotations_filtered:
            try: 
                fp = os.path.join(self.PROJECT_IMAGES_FP, a['StudyInstanceUID'], a['SeriesInstanceUID'], a['SOPInstanceUID']) + '.dcm'
            except Exception as error: 
                print('Exception:', error)
                print('a %s'% a)
            else: 
                image_fps.add(fp)
        return image_fps

    def get_local_image_annotations(self, image_fps, local_label_ids, annotations_filtered): 
        local_image_annotations = {fp: [] for fp in image_fps}
        for a in annotations_filtered:
            # only add local annotations with data
            if a['labelId'] in local_label_ids and a['data'] is not None:
                fp = os.path.join(self.PROJECT_IMAGES_FP, a['StudyInstanceUID'], a['SeriesInstanceUID'], a['SOPInstanceUID']) + '.dcm'
                local_image_annotations[fp].append(a)
        return local_image_annotations