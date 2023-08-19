import pandas as pd
from .common_utils import json_to_dataframe
from datetime import datetime
import os
import requests
import cv2
import numpy as np

import pydicom
from pydicom.filereader import dcmread
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.pixel_data_handlers.numpy_handler import pack_bits
from pydicom.sequence import Sequence
import warnings

warnings.filterwarnings("ignore", module="pydicom")


# Imports and Parse Some of the DICOM Standard Files
# -----------------------------------------------
class DicomExport:
    """
    Used to convert md.ai annotations to DICOM SR/SEG format for easier data processing.

    Inputs:
      `output_format` - determines if the output should be in DICOM SR/SEG (accepted inputs are "SR" or "SEG")
      `annotation_json` & `metadata_json` - are the exported annotation and metadata json paths from the md.ai project
                                                MAKE SURE THE DATASETS MATCH UP
      `combine_label_groups` - If `True` then each SEG file includes the annotations from every label group for that series
                               If `False` then a different SEG file will be created for each different label group annotation
                              (only applies to SEG output. Will be ignored for SR)
      `output_dir` - Specifies where the files should be downloaded
                     If None then files will be placed in a "SR(or SEG)_OUTPUT" folder in your cwd.

    Outputs:
      There will be a folder in your cwd, specified by the `output_dir` parameter, containing your SR/SEG exports.

    Created by Dyllan Hofflich and MD.ai.
    """

    def __init__(
        self,
        output_format,
        annotation_json,
        metadata_json,
        combine_label_groups=True,
        output_dir=None,
    ):
        self.Annotation_Json = annotation_json
        self.Metadata_Json = metadata_json
        self.output_format = output_format
        self.combine = combine_label_groups
        self.output_dir = output_dir

        if output_format != "SR" and output_format != "SEG":
            raise Exception('Invalid output format. Must be either "SR" or "SEG"')

        self.dicom_standards_setups()
        self.dicom_tags_setup()

    def dicom_standards_setups(self):
        """
        Searches the DICOM standard to gather which dicom tags are relevant to SR/Segmentation and which tags should be copied over from the original images.
        Gathers the DICOM standard data from https://github.com/innolitics/dicom-standard which is used in the Standard DICOM Browser website.
        """

        ctm_json = requests.get(
            "https://raw.githubusercontent.com/innolitics/dicom-standard/master/standard/ciod_to_modules.json"
        ).text
        mta_json = requests.get(
            "https://raw.githubusercontent.com/innolitics/dicom-standard/master/standard/module_to_attributes.json"
        ).text
        attributes_json = requests.get(
            "https://raw.githubusercontent.com/innolitics/dicom-standard/master/standard/attributes.json"
        ).text
        # ciod_to_modules_dataframe
        ctm_df = pd.read_json(ctm_json)
        # module_to_attributes_dataframe
        mta_df = pd.read_json(mta_json)
        # attributes_dataframe
        attributes_df = pd.read_json(attributes_json)

        # Select basic-text-sr/SEG modules
        if self.output_format == "SR":
            SR_modules_df = ctm_df[ctm_df["ciodId"] == "basic-text-sr"]
        else:
            SR_modules_df = ctm_df[ctm_df["ciodId"] == "segmentation"]
        # Select all basic-text-sr/SEG attributes
        SR_attributes_df = mta_df[mta_df["moduleId"].isin(SR_modules_df["moduleId"])]

        attribute_to_keyword_map = dict(zip(attributes_df["tag"], attributes_df["keyword"]))
        self.keyword_to_VR_map = dict(
            zip(attributes_df["keyword"], attributes_df["valueRepresentation"])
        )
        attribute_to_type_map = dict(zip(SR_attributes_df["tag"], SR_attributes_df["type"]))

        self.keyword_to_type_map = {}
        for attribute in attribute_to_type_map:
            self.keyword_to_type_map[attribute_to_keyword_map[attribute]] = attribute_to_type_map[
                attribute
            ]

        # Create dicom heirarchy for SR/SEG document (modeled after the Standard DICOM Browser)
        # ---------------------------------------------------
        SR_attributes_df.sort_values("path")
        self.dicom_tag_heirarchy = {}
        for _, row in SR_attributes_df.iterrows():
            if row["path"].count(":") == 1:
                self.dicom_tag_heirarchy[attribute_to_keyword_map[row["tag"]]] = {}
            else:
                paths = row["path"].split(":")
                # convert all tags in path to tag format
                parents = []
                for parent in paths[1:-1]:
                    parent = f"({parent[:4]},{parent[4:]})".upper()
                    parent = attribute_to_keyword_map[parent]
                    parents.append(parent)

                child = paths[-1]
                child = f"({child[:4]},{child[4:]})".upper()
                child = attribute_to_keyword_map[child]

                # get to last tag sequence
                current_sequence = self.dicom_tag_heirarchy[parents[0]]
                for parent in parents[1:]:
                    current_sequence = current_sequence[parent]
                current_sequence[child] = {}

        # Dictionary of VR and their corresponding types
        self.typos = {
            "AE": str,
            "AS": str,
            "AT": pydicom.tag.BaseTag,
            "CS": str,
            "DA": str,
            "DS": pydicom.valuerep.DSfloat,
            "DT": str,
            "FL": float,
            "FD": float,
            "IS": pydicom.valuerep.IS,
            "LO": str,
            "LT": str,
            "OB": bytes,
            "OB or OW": bytes,
            "OD": bytes,
            "OF": bytes,
            "OL": bytes,
            "OV": bytes,
            "OW": bytes,
            "PN": pydicom.valuerep.PersonName,
            "SH": str,
            "SL": int,
            "SQ": pydicom.sequence.Sequence,
            "SS": int,
            "ST": str,
            "SV": int,
            "TM": str,
            "UC": str,
            "UI": pydicom.uid.UID,
            "UL": int,
            "UN": bytes,
            "UR": str,
            "US": int,
            "US or SS": int,
            "UT": str,
            "UV": int,
        }

    def dicom_tags_setup(self):
        """
        Organizes the dicom tags and study, series, and image level information into a more parsable structure.
        """

        # Read Imported JSONs
        results = json_to_dataframe(os.getcwd() + "/" + self.Annotation_Json)
        self.metadata = pd.read_json(os.getcwd() + "/" + self.Metadata_Json)

        # Annotations dataframe
        self.annots_df = results["annotations"]
        labels = results["labels"]
        self.label_name_map = dict(zip(labels.labelId, labels.labelName))
        self.label_scope_map = dict(zip(labels.labelId, labels.scope))

        # Images DICOM Tags dataframe
        tags = []
        for dataset in self.metadata["datasets"]:
            tags.extend(dataset["dicomMetadata"])

        # Create organization of study, series, instance UID & dicom tags
        # ----------------------------------------------------------
        self.studies = self.annots_df.StudyInstanceUID.unique()
        self.tags_df = pd.DataFrame.from_dict(
            tags
        )  # dataframe of study, series, instance UID & dicom tags
        self.dicom_hierarchy = {}
        for tag in tags:
            study_uid = tag["StudyInstanceUID"]
            series_uid = tag["SeriesInstanceUID"]
            sop_uid = tag["SOPInstanceUID"]

            # Check if already seen study_uid yet (avoids key error)
            if study_uid not in self.dicom_hierarchy:  # Using study_uid bc rn it's exam level
                self.dicom_hierarchy[study_uid] = []

            # Dicom_heirarchy is a dictionary with study_uid as keys and a list as value
            # each list contains a dictionary with the series_uid as a key and a list of sop_uids as value
            if not any(series_uid in d for d in self.dicom_hierarchy[study_uid]):
                self.dicom_hierarchy[study_uid].append({series_uid: []})
            for d in self.dicom_hierarchy[
                study_uid
            ]:  # loops through item in dicom_heriarchy list (just the series_uid dict)
                if series_uid in d:
                    d[series_uid].append(sop_uid)

    # Helper functions to place DICOM tags into SR/SEG document Template
    # ---------------------------------------------------
    """
  > Iterates through a given sequence of tags from the standard DICOM heirarchy
  > Checks if the tag exists in the current DICOM file's headers
  >>  If it does then it adds the tag to the SR document dataset
  > Recursively calls itself to add tags in sequences and
  >>  Checks if a sequence contains all its required tags and adds them if so
  > Returns the SR document dataset with all tags added
  > If there were no tags added then returns False
  """

    def place_tags(self, dicom_tags, curr_dataset, curr_seq, need_to_check_required=True):
        sequences = {}
        added = False
        # Iterate through sequence to add tags and find sequences
        for keyword in curr_seq:
            if keyword in dicom_tags:
                curr_dataset = self.add_to_dataset(curr_dataset, keyword, dicom_tags[keyword], True)
                added = True
            if self.keyword_to_VR_map[keyword] == "SQ":
                sequences[keyword] = curr_seq[keyword]

        # Iterate through sequences to add tags and recursively search within sequences for tags
        for keyword in sequences:
            if (
                self.output_format == "SR" and keyword == "ContentSequence"
            ):  # Skips ContentSequence since it's meant to contain the annotations data
                continue
            seq = sequences[keyword]
            new_dataset = Dataset()
            new_dataset = self.place_tags(dicom_tags, new_dataset, seq, need_to_check_required)
            if new_dataset:
                if self.keyword_to_VR_map[keyword] == "SQ":
                    new_dataset = [new_dataset]  # Pydicom requires sequences to be in a list
                if not need_to_check_required or self.check_required(new_dataset, seq):
                    added = True
                    curr_dataset = self.add_to_dataset(curr_dataset, keyword, new_dataset, True)

        if added:
            return curr_dataset

        return False

    # Checks if a sequence contains all its required tags
    def check_required(self, curr_dataset, curr_seq):
        for keyword in curr_seq:
            tag_type = self.keyword_to_type_map[keyword]
            if keyword not in curr_dataset and "1" == tag_type:
                return False
        return True

    # Adds tag to dataset and if the tag already exists then
    # Replaces tag if replace=True if not then does nothing
    def add_to_dataset(self, dataset, keyword, value, replace):
        VR = self.keyword_to_VR_map[keyword]

        # If the tag is a sequence then the value in dicom_tags will be a list containing dictionary so need to convert to sequence format
        if type(value) == list and VR == "SQ":
            if type(value[0]) == dict:
                value = self.dict_to_sequence(value)

        # If the tag is a byte encoding then need to switch it to so from string
        if self.typos[VR] == bytes and value != None:
            value = value[2:-1].encode("UTF-8")  # removes b' and '

        # If the tag is an int/float encoding then need to switch it to so from string
        if self.typos[VR] == int or self.typos[VR] == float:
            if value != None:
                value = self.typos[VR](value)

        # check if tag already in dataset
        if keyword in dataset:
            if not replace:
                return dataset
            dataset[keyword].value = value
            return dataset

        if (
            "or SS" in VR and type(value) == int
        ):  # Fix bug when VR == 'US or SS' and the value is negative (it always defaults to US)
            if value < 0:
                VR = "SS"

        dataset.add_new(keyword, VR, value)
        return dataset

    # Creates a sequence from a list of dictionaries
    def dict_to_sequence(self, dict_seq_list):
        sequences = []
        for dict_seq in dict_seq_list:
            seq = Dataset()
            for keyword in dict_seq:
                if self.keyword_to_VR_map[keyword] == "SQ":
                    inner_seq = self.dict_to_sequence(dict_seq[keyword])
                    seq = self.add_to_dataset(seq, keyword, inner_seq, True)
                else:
                    seq = self.add_to_dataset(seq, keyword, dict_seq[keyword], True)
            sequences.append(seq)
        return sequences


class SrExport(DicomExport):
    def __init__(
        self,
        annotation_json,
        metadata_json,
        combine_label_groups=True,
        output_dir=None,
    ):
        DicomExport.__init__(
            self,
            "SR",
            annotation_json,
            metadata_json,
            combine_label_groups,
            output_dir,
        )
        self.create_sr_exports()

    def create_sr_exports(self):
        # Iterate through each study and create SR document for each annotator in each study
        # Save output to Output folder
        # ---------------------------------------------------
        try:
            if self.output_dir == None:
                out_dir = "SR_Output"
                os.mkdir("SR_Output")
            else:
                out_dir = self.output_dir
                os.mkdir(self.output_dir)
        except:
            pass

        from io import BytesIO

        document_file = os.path.join(os.path.dirname(__file__), "./sample_SR.dcm")
        for dataset_id in self.annots_df["datasetId"].unique():
            self.dataset_annots = self.annots_df[self.annots_df.datasetId == dataset_id]
            for study_uid in self.studies:
                # load file template
                ds = dcmread(document_file)

                self.dicom_tags = self.tags_df[
                    self.tags_df.StudyInstanceUID == study_uid
                ].dicomTags.values[0]
                annotations = self.dataset_annots[self.dataset_annots.StudyInstanceUID == study_uid]

                annotators = annotations.createdById.unique()
                series_uid = pydicom.uid.generate_uid(prefix=None)
                instance_uid = pydicom.uid.generate_uid(prefix=None)
                date = datetime.now().strftime("%Y%m%d")
                time = datetime.now().strftime("%H%M%S")

                # Place all the tags from the dicom into the SR document
                ds = self.place_tags(self.dicom_tags, ds, self.dicom_tag_heirarchy)

                # modify file metadata
                ds.file_meta.MediaStorageSOPInstanceUID = (
                    instance_uid  # Media Storage SOP Instance UID
                )
                ds.file_meta.ImplementationClassUID = str(
                    pydicom.uid.PYDICOM_IMPLEMENTATION_UID
                )  # Implementation Class UID
                ds.file_meta.ImplementationVersionName = str(
                    pydicom.__version__
                )  # Implementation Version Name

                # delete tags
                del ds[0x00080012]  # Instance Creation Date
                del ds[0x00080013]  # Instance Creation Time
                del ds[0x00080014]  # Instance Creator UID
                # del ds[0x00100030]  # Patient's Birth Date

                # modify tags
                # -------------------------

                ds[
                    "SOPClassUID"
                ].value = "1.2.840.10008.5.1.4.1.1.88.22"  # SOP Class UID = enhanced SR storage
                ds[0x00080018].value = instance_uid  # SOPInstanceUID
                ds[0x0008103E].value = str(self.metadata["name"].values[0])  # Series Description
                ds[0x00080021].value = str(date)  # Series Date
                ds[0x00080023].value = str(date)  # Content Date
                ds[0x00080031].value = str(time)  # Series Time
                ds[0x00080033].value = str(time)  # Content Time

                ds[0x00181020].value = ""  # Software Versions

                ds[0x0020000D].value = str(study_uid)  # Study Instance UID
                ds[0x0020000E].value = str(series_uid)  # Series Instance UID
                ds[0x00200011].value = str(1)  # Series Number

                ds.Modality = "SR"

                # create dicom hierarchy
                dicom_hier = self.dicom_hierarchy[study_uid]
                series_sequence = []
                for series in dicom_hier:
                    for key in series:
                        sops = series[key]
                        series_hier = Dataset()
                        sop_sequence = []
                        for sop in sops:
                            sop_data = Dataset()
                            if "SOPClassUID" in self.dicom_tags:
                                sop_data.ReferencedSOPClassUID = self.dicom_tags["SOPClassUID"]
                            sop_data.ReferencedSOPInstanceUID = sop
                            sop_sequence.append(sop_data)
                        series_hier.ReferencedSOPSequence = sop_sequence
                        series_hier.SeriesInstanceUID = key
                        series_sequence.append(series_hier)

                ds[0x0040A375][0].ReferencedSeriesSequence = series_sequence
                ds[0x0040A375][0].StudyInstanceUID = study_uid

                # add tags
                ds[0x00080005] = pydicom.dataelem.DataElement(
                    0x00080005, "CS", "ISO_IR 192"
                )  # Specific Character Set

                # create content for each annotator
                for i in range(len(annotators)):
                    instance_number = i + 1
                    ds[0x00200013] = pydicom.dataelem.DataElement(
                        0x00200013, "IS", str(instance_number)
                    )  # Instance Number
                    ds[0x0040A730][0][0x0040A123].value = f"Annotator{instance_number}"
                    ds[0x0040A078][0][0x0040A123].value = f"Annotator{instance_number}"
                    anns = annotations[annotations.createdById == annotators[i]]

                    anns_map = {}

                    def annotator_iteration(row):
                        annotation = []
                        label_id = row["labelId"]
                        parent_id = row["parentLabelId"]
                        annotation.extend(
                            [
                                parent_id,
                                row["scope"],
                                row["SOPInstanceUID"],
                                row["SeriesInstanceUID"],
                            ]
                        )
                        if "SOPClassUID" in self.dicom_tags:
                            annotation.append(self.dicom_tags["SOPClassUID"])

                        if label_id not in anns_map:
                            anns_map[label_id] = []
                        anns_map[label_id].append(annotation)

                    anns.apply(annotator_iteration, axis=1)

                    # annotator_iteration has extraneous labels for those with child labels as it creates 2 separate entries for the child label and the parent label
                    for label_id in anns_map:
                        for annot in anns_map[label_id]:
                            if annot[0] != None:
                                if (
                                    annot[0] not in anns_map
                                ):  # Fixes edge case where a child label appears with no parent label for that annotator
                                    continue  # Occurs when another annotator adds a child label to a different annotator's label

                                for j in range(
                                    len(anns_map[annot[0]]) - 1, -1, -1
                                ):  # iterate backwards so can delete while iterating
                                    parent_annot = anns_map[annot[0]][j]
                                    if (
                                        (
                                            type(parent_annot[2]) == type(annot[2])
                                            and type(annot[2] == float)
                                        )
                                        and (
                                            type(parent_annot[3]) == type(annot[3])
                                            and type(annot[3] == float)
                                        )
                                    ) or (
                                        (parent_annot[2] == annot[2])
                                        and (parent_annot[3] == annot[3])
                                    ):  # check if series and sop uid are same
                                        del anns_map[annot[0]][j]

                    content_sequence = []
                    code_number = 43770  # hello

                    # Create a list of labelIds ordered from exam to series to image
                    ordered_labels = []
                    j = 0
                    for label_id in anns_map:
                        if self.label_scope_map[label_id] == "EXAM":
                            ordered_labels.insert(0, label_id)
                            j += 1
                        elif self.label_scope_map[label_id] == "INSTANCE":
                            ordered_labels.append(label_id)
                        else:
                            ordered_labels.insert(j, label_id)

                    for label_id in ordered_labels:
                        for a in anns_map[label_id]:
                            # Add 'Referenced Segment' if label is in IMAGE scope
                            if a[1] == "INSTANCE":
                                content = Dataset()
                                content.ValueType = "IMAGE"
                                referenced_sequence_ds = Dataset()
                                if len(a) > 4:
                                    referenced_sequence_ds.ReferencedSOPClassUID = a[4]
                                referenced_sequence_ds.ReferencedSOPInstanceUID = a[2]
                                content.ReferencedSOPSequence = [referenced_sequence_ds]

                                code_sequence_ds = Dataset()
                                code_sequence_ds.CodeValue = str(code_number)
                                code_sequence_ds.CodingSchemeDesignator = "99MDAI"
                                code_sequence_ds.CodeMeaning = "Referenced Image"
                                code_sequence = [code_sequence_ds]
                                content.ConceptNameCodeSequence = code_sequence
                                code_number += 1
                                content_sequence.append(content)

                            # Add parent label to text value
                            content = Dataset()
                            code_sequence_ds = Dataset()
                            if a[0] != None:
                                code_name = self.label_name_map[a[0]]
                            else:
                                code_name = self.label_name_map[label_id]
                            code_sequence_ds.CodeValue = str(hash(code_name))[1:6]
                            code_sequence_ds.CodingSchemeDesignator = "99MDAI"
                            code_sequence_ds.CodeMeaning = code_name
                            code_sequence = [code_sequence_ds]
                            content.ConceptNameCodeSequence = code_sequence

                            # Add child label text
                            text_value = ""
                            if a[0] != None:
                                text_value = ",".join(
                                    map(lambda labelId: self.label_name_map[labelId], [label_id])
                                )
                                text_value += "\n"
                                content.TextValue = text_value
                            # Add 'Series UID:'
                            if a[1] == "SERIES":
                                text_value += f"Series UID: {series_uid}"
                                content.TextValue = text_value
                            if text_value != "":
                                content.ValueType = "TEXT"
                            else:
                                content.ValueType = "CONTAINER"
                            content_sequence.append(content)

                    ds[0x0040A730][1][0x0040A730][0].ContentSequence = content_sequence

                    ds.save_as(
                        f"{os.getcwd()}/{out_dir}/DICOM_SR_{dataset_id}_{study_uid}_annotator_{instance_number}.dcm"
                    )
        print(f"Successfully exported DICOM SR files into {out_dir}")


class SegExport(DicomExport):
    def __init__(
        self,
        annotation_json,
        metadata_json,
        combine_label_groups=True,
        output_dir=None,
    ):
        DicomExport.__init__(
            self,
            "SEG",
            annotation_json,
            metadata_json,
            combine_label_groups,
            output_dir,
        )
        self.create_seg_exports()

    # Annotation dataframe has a separate row for a parent label. This function drops that row
    def drop_dupes(self, row):
        if row["parentLabelId"] != None:
            if (
                row["parentLabelId"] not in self.annots_df["labelId"].unique()
            ):  # Fixes edge case where a child label appears with no parent label for that annotator
                return  # Occurs when another annotator adds a child label to a different annotator's label
            parents = self.annots_df[self.annots_df["labelId"] == row["parentLabelId"]]
            study_parents = parents[parents["StudyInstanceUID"] == row["StudyInstanceUID"]]
            series_parents = study_parents[
                study_parents["SeriesInstanceUID"] == row["SeriesInstanceUID"]
            ]
            sop_parents = series_parents[series_parents["SOPInstanceUID"] == row["SOPInstanceUID"]]

            if len(sop_parents.index) > 0:
                self.annots_df.drop(sop_parents.index[0], inplace=True)
            elif len(series_parents.index) > 0:
                self.annots_df.drop(series_parents.index[0], inplace=True)
            elif len(study_parents.index) > 0:
                self.annots_df.drop(study_parents.index[0], inplace=True)

    # Gets imgs from annotations and creates segment sequence

    def img_insert(self, row, ds):
        data = self.load_mask_instance(row)
        if not np.isscalar(data):
            if self.prev_annot is not None and (
                self.prev_annot["labelId"] == row["labelId"]
                and self.prev_annot["labelGroupName"] == row["labelGroupName"]
                and self.prev_annot["instanceNumber"] == row["instanceNumber"]
            ):
                mask2 = self.load_mask_instance(row)
                self.imgs[-1] = np.ma.mask_or(self.imgs[-1], mask2)
            else:
                self.imgs.append(self.load_mask_instance(row))
                self.included_sops.append((len(self.seen_labels) + 1, row["SOPInstanceUID"]))
                self.unique_sops.add(row["SOPInstanceUID"])
                self.name_number_map[len(self.seen_labels) + 1] = row["labelName"]
            self.prev_annot = row

            if row["labelId"] not in self.seen_labels:
                if row["parentLabelId"] == None:
                    parent_label_name = self.label_name_map[row["labelId"]]
                else:
                    parent_label_name = self.label_name_map[row["parentLabelId"]]
                child_label_name = self.label_name_map[row["labelId"]]

                segment_sequence = ds.SegmentSequence

                segment1 = Dataset()
                segment_sequence.append(segment1)

                # Segmented Property Category Code Sequence
                segmented_property_category_code_sequence = Sequence()
                segment1.SegmentedPropertyCategoryCodeSequence = (
                    segmented_property_category_code_sequence
                )

                # Segmented Property Category Code Sequence: Segmented Property Category Code 1
                segmented_property_category_code1 = Dataset()
                segmented_property_category_code_sequence.append(segmented_property_category_code1)
                segmented_property_category_code1.CodeValue = str(hash(parent_label_name))[1:6]
                segmented_property_category_code1.CodingSchemeDesignator = "99MDAI"
                segmented_property_category_code1.CodeMeaning = (
                    f'{parent_label_name} from Label Group {row["labelGroupName"]}'
                )

                segment1.SegmentNumber = len(self.seen_labels) + 1  # (number of labels)
                segment1.SegmentLabel = child_label_name
                segment1.SegmentAlgorithmType = "MANUAL"  # Maybe change based on how it was created

                # Segmented Property Type Code Sequence
                segmented_property_type_code_sequence = Sequence()
                segment1.SegmentedPropertyTypeCodeSequence = segmented_property_type_code_sequence

                # Segmented Property Type Code Sequence: Segmented Property Type Code 1
                segmented_property_type_code1 = Dataset()
                segmented_property_type_code_sequence.append(segmented_property_type_code1)
                segmented_property_type_code1.CodeValue = str(hash(child_label_name))[1:6]
                segmented_property_type_code1.CodingSchemeDesignator = "99MDAI"
                segmented_property_type_code1.CodeMeaning = child_label_name

                self.seen_labels.add(row["labelId"])

    def load_mask_instance(self, row):
        """Load instance masks for the given annotation row. Masks can be different types,
        mask is a binary true/false map of the same size as the image.
        """

        if row.data == None:
            return 404  # no data found

        mask = np.zeros((int(row.height), int(row.width)), dtype=np.uint8)

        annotation_mode = row.annotationMode

        if annotation_mode == "bbox":
            # Bounding Box
            x = int(row.data["x"])
            y = int(row.data["y"])
            w = int(row.data["width"])
            h = int(row.data["height"])
            mask_instance = mask[:, :].copy()
            cv2.rectangle(mask_instance, (x, y), (x + w, y + h), 255, -1)
            mask[:, :] = mask_instance

        # FreeForm or Polygon
        elif annotation_mode == "freeform" or annotation_mode == "polygon":
            vertices = np.array(row.data["vertices"])
            vertices = vertices.reshape((-1, 2))
            mask_instance = mask[:, :].copy()
            cv2.fillPoly(mask_instance, np.int32([vertices]), (255, 255, 255))
            mask[:, :] = mask_instance

        # Line
        elif annotation_mode == "line":
            vertices = np.array(row.data["vertices"])
            vertices = vertices.reshape((-1, 2))
            mask_instance = mask[:, :].copy()
            cv2.polylines(mask_instance, np.int32([vertices]), False, (255, 255, 255), 12)
            mask[:, :] = mask_instance

        elif annotation_mode == "location":
            # Bounding Box
            x = int(row.data["x"])
            y = int(row.data["y"])
            mask_instance = mask[:, :].copy()
            cv2.circle(mask_instance, (x, y), 7, (255, 255, 255), -1)
            mask[:, :] = mask_instance

        elif annotation_mode == "ellipse":
            cx = int(row.data["cx"])
            cy = int(row.data["cy"])
            rx = int(row.data["rx"])
            ry = int(row.data["ry"])
            mask_instance = mask[:, :].copy()
            cv2.ellipse(mask_instance, (cx, cy), (rx, ry), 0, 0, 360, (255, 255, 255), 12)
            mask[:, :] = mask_instance

        elif annotation_mode == "mask":
            mask_instance = mask[:, :].copy()
            if row.data["foreground"]:
                for i in row.data["foreground"]:
                    mask_instance = cv2.fillPoly(
                        mask_instance, [np.array(i, dtype=np.int32)], (255, 255, 255)
                    )
            if row.data["background"]:
                for i in row.data["background"]:
                    mask_instance = cv2.fillPoly(
                        mask_instance, [np.array(i, dtype=np.int32)], (0, 0, 0)
                    )
            mask[:, :] = mask_instance

        return mask.astype(bool)

    def create_seg_exports(self):
        """
        Creates a template SEG File and adds in necessary SEG information
        Instead of working from a template, this function creates a segmentation file from scratch using pydicom
        """
        try:
            if self.output_dir == None:
                out_dir = "SEG_Output"
                os.mkdir("SEG_Output")
            else:
                out_dir = self.output_dir
                os.mkdir(self.output_dir)
        except:
            pass

        self.annots_df.apply(self.drop_dupes, axis=1)

        for dataset_id in self.annots_df["datasetId"].unique():
            self.dataset_annots = self.annots_df[self.annots_df.datasetId == dataset_id]
            for study_uid in self.studies:
                dicom_hier = self.dicom_hierarchy[study_uid]
                for series_dict in dicom_hier:
                    for series_uid in series_dict:
                        sops = series_dict[series_uid]

                        annotations = self.dataset_annots[
                            self.dataset_annots.SeriesInstanceUID == series_uid
                        ]
                        annotations = annotations[annotations["scope"] == "INSTANCE"]
                        if annotations.empty:
                            continue

                        self.dicom_tags = self.tags_df[
                            self.tags_df.SeriesInstanceUID == series_uid
                        ].dicomTags.values[0]
                        annotators = annotations.createdById.unique()
                        instance_uid = pydicom.uid.generate_uid(prefix=None)
                        date = datetime.now().strftime("%Y%m%d")
                        time = datetime.now().strftime("%H%M%S")

                        sop_instance_num_map = {}
                        for sop in sops:
                            sop_dicom_tags = self.tags_df[
                                self.tags_df.SOPInstanceUID == sop
                            ].dicomTags.values[0]
                            if "InstanceNumber" in sop_dicom_tags:
                                sop_instance_num_map[sop] = sop_dicom_tags["InstanceNumber"]
                            else:
                                sop_instance_num_map[sop] = "1"

                        def create_instance_number(row):
                            return sop_instance_num_map[row["SOPInstanceUID"]]

                        try:
                            annotations["instanceNumber"] = annotations.apply(
                                create_instance_number, axis=1
                            )
                        except:
                            continue
                        annotations = annotations.sort_values(
                            ["labelGroupName", "labelId", "instanceNumber"], ignore_index=True
                        )  # sort by label group then annotation then appearance in series

                        # File meta info data elements
                        file_meta = FileMetaDataset()
                        file_meta.FileMetaInformationVersion = b"\x00\x01"
                        file_meta.TransferSyntaxUID = "1.2.840.10008.1.2.1"
                        file_meta.MediaStorageSOPInstanceUID = instance_uid  # Create Instance UID      # Media Storage SOP Instance UID
                        file_meta.ImplementationClassUID = str(
                            pydicom.uid.PYDICOM_IMPLEMENTATION_UID
                        )  # Implementation Class UID
                        file_meta.ImplementationVersionName = str(
                            pydicom.__version__
                        )  # Implementation Version Name
                        file_meta.SourceApplicationEntityTitle = "POSDA"

                        # Main data elements
                        ds = Dataset()

                        ds = self.place_tags(self.dicom_tags, ds, self.dicom_tag_heirarchy, True)

                        ds.SpecificCharacterSet = "ISO_IR 192"
                        ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.66.4"
                        ds.SOPInstanceUID = instance_uid
                        ds.SeriesDate = str(date)  # Series Date
                        ds.ContentDate = str(date)  # Content Date
                        ds.SeriesTime = str(time)  # Series Time
                        ds.ContentTime = str(time)  # Series Time
                        ds.Manufacturer = "MDAI"
                        ds.Modality = "SEG"

                        # Referenced Series Sequence
                        refd_series_sequence = Sequence()
                        ds.ReferencedSeriesSequence = refd_series_sequence

                        # Referenced Series Sequence: Referenced Series 1
                        refd_series1 = Dataset()
                        refd_series_sequence.append(refd_series1)

                        # Referenced Series Sequence: Referenced Series 1
                        refd_series1 = Dataset()
                        refd_series_sequence.append(refd_series1)
                        refd_series1.SeriesInstanceUID = series_uid

                        # Referenced Instance Sequence
                        refd_instance_sequence = Sequence()
                        refd_series1.ReferencedInstanceSequence = refd_instance_sequence

                        ds.SegmentationType = "BINARY"

                        for annotator_id in annotators:
                            annotator_annots = annotations[annotations.createdById == annotator_id]

                            if self.combine:
                                label_group_sets = [annotator_annots.labelGroupName.unique()]
                            else:
                                label_group_sets = [
                                    [group] for group in annotator_annots.labelGroupName.unique()
                                ]

                            ds.SamplesPerPixel = 1
                            ds.PhotometricInterpretation = "MONOCHROME2"
                            ds.BitsAllocated = 1
                            ds.BitsStored = 1
                            ds.HighBit = 0
                            ds.PixelRepresentation = 0
                            ds.LossyImageCompression = "00"

                            for label_group_set in label_group_sets:
                                label_group_annots = annotator_annots[
                                    annotator_annots.labelGroupName.isin(label_group_set)
                                ]

                                # Segment Sequence
                                segment_sequence = Sequence()
                                ds.SegmentSequence = segment_sequence

                                self.imgs = []
                                self.seen_labels = set()
                                self.name_number_map = {}
                                self.included_sops = []
                                self.unique_sops = set()
                                self.label_groups = list(annotations.labelGroupName.unique())
                                self.prev_annot = None
                                label_group_annots.apply(self.img_insert, args=(ds,), axis=1)

                                ds.NumberOfFrames = len(
                                    self.imgs
                                )  # create during last parts of SEG file (should equal length of annot_df)
                                ds.PixelData = pack_bits(np.array(self.imgs))

                                for sop in self.unique_sops:
                                    sop_dicom_tags = self.tags_df[
                                        self.tags_df.SOPInstanceUID == sop
                                    ].dicomTags.values[0]
                                    refd_instance1 = Dataset()
                                    refd_instance_sequence.append(refd_instance1)
                                    if "SOPClassUID" in sop_dicom_tags:
                                        refd_instance1.ReferencedSOPClassUID = sop_dicom_tags[
                                            "SOPClassUID"
                                        ]
                                    refd_instance1.ReferencedSOPInstanceUID = sop_dicom_tags[
                                        "SOPInstanceUID"
                                    ]

                                # Leaving it out for now but if nothing works then maybe try to add it back in blank and then with dummy values
                                # Edit: added it back in but still unnecessary.
                                # -----------------------------------------------------------------------
                                # Dimension Index Sequence
                                dimension_index_sequence = Sequence()
                                ds.DimensionIndexSequence = dimension_index_sequence
                                # -----------------------------------------------------------------------

                                ds.ContentLabel = "MDAI_SEG"
                                ds.ContentCreatorName = f"annotator {annotator_id}"

                                # Leaving it out for now but if nothing works then maybe try to add it back in blank and then with dummy values
                                # Edit: added it back in but still unnecessary.
                                # -----------------------------------------------------------------------
                                # Shared Functional Groups Sequence
                                shared_functional_groups_sequence = Sequence()
                                ds.SharedFunctionalGroupsSequence = (
                                    shared_functional_groups_sequence
                                )
                                # -----------------------------------------------------------------------

                                # Per-frame Functional Groups Sequence
                                per_frame_functional_groups_sequence = Sequence()
                                ds.PerFrameFunctionalGroupsSequence = (
                                    per_frame_functional_groups_sequence
                                )

                                # Per-frame Functional Groups Sequence
                                per_frame_functional_groups_sequence = Sequence()
                                ds.PerFrameFunctionalGroupsSequence = (
                                    per_frame_functional_groups_sequence
                                )

                                # Per-frame Functional Groups Sequence
                                per_frame_functional_groups_sequence = []

                                # Loop through each frame with an annotation and create unique Per Frame Functional Group Sequence
                                # ---------------------------------------------------------------------------
                                for segment_number, sop in self.included_sops:
                                    label_names = ", ".join(
                                        label_group_annots["labelName"].unique()
                                    )
                                    ds.SeriesDescription = (
                                        f"Segmentation of {label_names} by annotator {annotator_id}"
                                    )

                                    sop_dicom_tags = self.tags_df[
                                        self.tags_df.SOPInstanceUID == sop
                                    ].dicomTags.values[0]

                                    # Per-frame Functional Groups Sequence: Per-frame Functional Groups 1
                                    per_frame_functional_groups1 = Dataset()
                                    per_frame_functional_groups_sequence.append(
                                        per_frame_functional_groups1
                                    )

                                    # Derivation Image Sequence
                                    derivation_image_sequence = Sequence()
                                    per_frame_functional_groups1.DerivationImageSequence = (
                                        derivation_image_sequence
                                    )

                                    # Derivation Image Sequence: Derivation Image 1
                                    derivation_image1 = Dataset()
                                    derivation_image_sequence.append(derivation_image1)

                                    # Source Image Sequence
                                    source_image_sequence = Sequence()
                                    derivation_image1.SourceImageSequence = source_image_sequence

                                    # Source Image Sequence: Source Image 1
                                    source_image1 = Dataset()
                                    source_image_sequence.append(source_image1)
                                    if "SOPClassUID" in self.dicom_tags:
                                        source_image1.ReferencedSOPClassUID = self.dicom_tags[
                                            "SOPClassUID"
                                        ]
                                    source_image1.ReferencedSOPInstanceUID = self.dicom_tags[
                                        "SOPInstanceUID"
                                    ]

                                    # Purpose of Reference Code Sequence
                                    purpose_of_ref_code_sequence = Sequence()
                                    source_image1.PurposeOfReferenceCodeSequence = (
                                        purpose_of_ref_code_sequence
                                    )

                                    # Purpose of Reference Code Sequence: Purpose of Reference Code 1
                                    purpose_of_ref_code1 = Dataset()
                                    purpose_of_ref_code_sequence.append(purpose_of_ref_code1)
                                    purpose_of_ref_code1.CodeValue = "121322"
                                    purpose_of_ref_code1.CodingSchemeDesignator = "DCM"
                                    purpose_of_ref_code1.CodeMeaning = (
                                        "Source image for image processing operation"
                                    )

                                    # Derivation Code Sequence
                                    derivation_code_sequence = Sequence()
                                    derivation_image1.DerivationCodeSequence = (
                                        derivation_code_sequence
                                    )

                                    # Derivation Code Sequence: Derivation Code 1
                                    derivation_code1 = Dataset()
                                    derivation_code_sequence.append(derivation_code1)
                                    derivation_code1.CodeValue = "113076"
                                    derivation_code1.CodingSchemeDesignator = "DCM"
                                    derivation_code1.CodeMeaning = "Segmentation"

                                    # Segment Identification Sequence
                                    segment_id_seq = Dataset()
                                    per_frame_functional_groups1.SegmentIdentificationSequence = [
                                        segment_id_seq
                                    ]

                                    # Segment Number
                                    segment_id_seq.ReferencedSegmentNumber = segment_number

                                    per_frame_functional_groups1 = self.place_tags(
                                        sop_dicom_tags,
                                        per_frame_functional_groups1,
                                        self.dicom_tag_heirarchy[
                                            "PerFrameFunctionalGroupsSequence"
                                        ],
                                        False,
                                    )
                                # -------------------------------------------------------------------------

                                ds.PerFrameFunctionalGroupsSequence = (
                                    per_frame_functional_groups_sequence
                                )

                                ds.file_meta = file_meta
                                ds.is_implicit_VR = False
                                ds.is_little_endian = True

                                if self.included_sops:
                                    if self.combine:
                                        ds.save_as(
                                            f"{os.getcwd()}/{out_dir}/DICOM_SEG_{dataset_id}_{series_uid}_annotator_{annotator_id}.dcm",
                                            False,
                                        )
                                    else:
                                        ds.save_as(
                                            f"{os.getcwd()}/{out_dir}/DICOM_SEG_{dataset_id}_label_group_{label_group_set[0]}_series_{series_uid}_annotator_{annotator_id}.dcm",
                                            False,
                                        )
        print(f"Successfully exported DICOM SEG files into {out_dir}")


def iterate_content_seq(content, content_seq_list):
    """
    util helper function iterating through DICOM-SR content sequences and append to list
    """
    for content_seq in content_seq_list:
        parent_labels = []
        child_labels = []
        notes = []

        if "RelationshipType" in content_seq:
            if content_seq.RelationshipType == "HAS ACQ CONTEXT":
                continue

        if content_seq.ValueType == "IMAGE":
            if "ReferencedSOPSequence" in content_seq:
                for ref_seq in content_seq.ReferencedSOPSequence:
                    if "ReferencedSOPClassUID" in ref_seq:
                        notes.append(
                            f"\n   Referenced SOP Class UID = {ref_seq.ReferencedSOPClassUID}"
                        )
                    if "ReferencedSOPInstanceUID" in ref_seq:
                        notes.append(
                            f"\n   Referenced SOP Instance UID = {ref_seq.ReferencedSOPInstanceUID}"
                        )
                    if "ReferencedSegmentNumber" in ref_seq:
                        notes.append(
                            f"\n   Referenced Segment Number = {ref_seq.ReferencedSegmentNumber}"
                        )
            else:
                continue

        if "ConceptNameCodeSequence" in content_seq:
            if len(content_seq.ConceptNameCodeSequence) > 0:
                parent_labels.append(content_seq.ConceptNameCodeSequence[0].CodeMeaning)
        if "ConceptCodeSequence" in content_seq:
            if len(content_seq.ConceptCodeSequence) > 0:
                child_labels.append(content_seq.ConceptCodeSequence[0].CodeMeaning)

        if "DateTime" in content_seq:
            notes.append(content_seq.DateTime)
        if "Date" in content_seq:
            notes.append(content_seq.Date)
        if "PersonName" in content_seq:
            notes.append(str(content_seq.PersonName))
        if "UID" in content_seq:
            notes.append(content_seq.UID)
        if "TextValue" in content_seq:
            child_labels.append(content_seq.TextValue)
        if "MeasuredValueSequence" in content_seq:
            if len(content_seq.MeasuredValueSequence) > 0:
                units = (
                    content_seq.MeasuredValueSequence[0].MeasurementUnitsCodeSequence[0].CodeValue
                )
                notes.append(str(content_seq.MeasuredValueSequence[0].NumericValue) + units)

        if "ContentSequence" in content_seq:
            iterate_content_seq(content, list(content_seq.ContentSequence))
        else:
            content.append([", ".join(parent_labels), ", ".join(child_labels), ", ".join(notes)])
