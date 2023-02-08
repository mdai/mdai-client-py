import random
import copy
import json
import pandas as pd

import os
import uuid
from functools import partial
import multiprocessing
import pydicom
import numpy as np
import nibabel as nib
from tqdm import tqdm
import cv2


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
    train_dataset.name = dataset.name + "-TRAIN"

    valid_dataset = copy.deepcopy(dataset)
    valid_dataset.id = dataset.id + "-VALID"
    valid_dataset.name = dataset.name + "-VALID"

    imgs_anns_dict = dataset.imgs_anns_dict

    train_imgs_anns_dict = filter_by_ids(train_image_ids, imgs_anns_dict)
    valid_imgs_anns_dict = filter_by_ids(valid_image_ids, imgs_anns_dict)

    train_dataset.image_ids = train_image_ids
    valid_dataset.image_ids = valid_image_ids

    train_dataset.imgs_anns_dict = train_imgs_anns_dict
    valid_dataset.imgs_anns_dict = valid_imgs_anns_dict

    all_train_annotations = []
    for _, annotations in train_dataset.imgs_anns_dict.items():
        all_train_annotations += annotations
    train_dataset.all_annotations = all_train_annotations

    all_val_annotations = []
    for _, annotations in valid_dataset.imgs_anns_dict.items():
        all_val_annotations += annotations
    valid_dataset.all_annotations = all_val_annotations

    print(
        "Num of instances for training set: %d, validation set: %d"
        % (len(train_image_ids), len(valid_image_ids))
    )
    return train_dataset, valid_dataset


def json_to_dataframe(json_file, datasets=[]):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    a = pd.DataFrame([])
    studies = pd.DataFrame([])
    labels = None

    # Gets annotations for all datasets
    for d in data["datasets"]:
        if d["id"] in datasets or len(datasets) == 0:
            study = pd.DataFrame(d["studies"])
            study["dataset"] = d["name"]
            study["datasetId"] = d["id"]
            studies = pd.concat([studies, study], ignore_index=True, sort=False)

            annots = pd.DataFrame(d["annotations"])
            annots["dataset"] = d["name"]
            a = pd.concat([a, annots], ignore_index=True, sort=False)

    if len(studies) > 0:
        studies = studies[["StudyInstanceUID", "dataset", "datasetId", "number"]]
    g = pd.DataFrame(data["labelGroups"])
    # unpack arrays
    result = pd.DataFrame([(d, tup.id, tup.name) for tup in g.itertuples() for d in tup.labels])
    if len(result) > 0:
        result.columns = ["labels", "labelGroupId", "labelGroupName"]

        def unpack_dictionary(df, column):
            ret = None
            ret = pd.concat(
                [df, pd.DataFrame((d for idx, d in df[column].iteritems()))], axis=1, sort=False
            )
            del ret[column]
            return ret

        labels = unpack_dictionary(result, "labels")
        if "parentId" in labels.columns:
            labels = labels[
                [
                    "labelGroupId",
                    "labelGroupName",
                    "annotationMode",
                    "color",
                    "description",
                    "id",
                    "name",
                    "radlexTagIds",
                    "scope",
                    "parentId",
                ]
            ]
            labels.columns = [
                "labelGroupId",
                "labelGroupName",
                "annotationMode",
                "color",
                "description",
                "labelId",
                "labelName",
                "radlexTagIdsLabel",
                "scope",
                "parentLabelId",
            ]
        else:
            labels = labels[
                [
                    "labelGroupId",
                    "labelGroupName",
                    "annotationMode",
                    "color",
                    "description",
                    "id",
                    "name",
                    "radlexTagIds",
                    "scope",
                ]
            ]
            labels.columns = [
                "labelGroupId",
                "labelGroupName",
                "annotationMode",
                "color",
                "description",
                "labelId",
                "labelName",
                "radlexTagIdsLabel",
                "scope",
            ]

        if len(a) > 0:
            a = a.merge(labels, on=["labelId"], sort=False)
    if len(studies) > 0 and len(a) > 0:
        a = a.merge(studies, on=["StudyInstanceUID", "dataset"], sort=False)
        # Format data
        studies.number = studies.number.astype(int)
        a.number = a.number.astype(int)
        a.loc.createdAt = pd.to_datetime(a.createdAt)
        a.loc.updatedAt = pd.to_datetime(a.updatedAt)
    return {"annotations": a, "studies": studies, "labels": labels}


def convert_mask_annotation_to_array(row):
    """
    Converts a dataframe row containing a mask annotation from our internal complex polygon data representation to a numpy array.
    """
    mask = np.zeros((int(row.width), int(row.height)))
    if row.data["foreground"]:
        for i in row.data["foreground"]:
            mask = cv2.fillPoly(mask, [np.array(i, dtype=np.int32)], 1)
    if row.data["background"]:
        for i in row.data["background"]:
            mask = cv2.fillPoly(mask, [np.array(i, dtype=np.int32)], 0)
    return mask


def convert_mask_data(data):
    """
    Converts a numpy array mask to our internal complex polygon data representation.
    """
    mask = np.uint8(np.array(data) > 0)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = [contours[i].reshape(-1, 2) for i in range(len(contours))]

    # Separate contours based on foreground / background polygons
    output_data = {
        "foreground": [],
        "background": [],
    }

    counts = [0] * len(contours)
    for i in range(len(contours)):
        parent = hierarchy[0][i][-1]
        if parent != -1:
            counts[i] = counts[parent] + 1

        if counts[i] % 2:
            output_data["background"].append(contours[i].tolist())
        else:
            output_data["foreground"].append(contours[i].tolist())
    return output_data


"""Converts NIFTI format to DICOM for CT exams. MR to come...

"""


def convert_ct(
    input_dir=None,
    output_dir=None,
    input_ext=".nii.gz",
    plane="axial",
    sample_dicom_fp=os.path.join(os.path.dirname(""), "./sample_dicom.dcm"),
    window_center=40,
    window_width=350,
):
    if not os.path.exists(input_dir):
        raise IOError("{:s} does not exist.".format(input_dir))
    if not os.path.exists(sample_dicom_fp):
        raise IOError("{:s} does not exist.".format(sample_dicom_fp))
    if plane not in ["axial", "sagittal", "coronal"]:
        raise ValueError("`plane` must be one of axial, sagittal, or coronal.")

    # make output dir if doesn't already exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    found_filepaths = list(_get_files(input_dir, ext=input_ext))
    print(f"{len(found_filepaths)} *{input_ext} files found. Processing...")

    n_procs = multiprocessing.cpu_count() - 1
    with multiprocessing.Pool(n_procs) as p:
        kwargs = {
            "input_dir": input_dir,
            "output_dir": output_dir,
            "input_ext": input_ext,
            "plane": plane,
            "sample_dicom_fp": sample_dicom_fp,
            "window_center": window_center,
            "window_width": window_width,
        }
        # need to iterate since Pool.imap is lazy
        for n in tqdm(
            p.imap_unordered(partial(_convert_nii_file_ct, **kwargs), found_filepaths),
            total=len(found_filepaths),
        ):
            pass


def _get_files(root, ext=None):
    """Yields all file paths recursively from root path, optionally filtering on extension.
    """
    for item in os.scandir(root):
        if item.is_file():
            if not ext:
                yield item.path
            elif os.path.splitext(item.path)[1] == ext or item.path.endswith(ext):
                yield item.path
        elif item.is_dir():
            yield from _get_files(item.path)


def _get_datatype(headers):
    dt = str(headers.get_data_dtype())
    return np.int16


# header datatype not reliable

#     if dt == 'int8':
#         return np.int8
#     elif dt == 'int16':
#         return np.int16
#     elif dt == 'int32':
#         return np.int32
#     elif dt == 'int64':
#         return np.int64
#     elif dt == 'float32':
#         return np.float32
#     elif dt == 'float64':
#         return np.float64
#     return np.int16


def _convert_nii_file_ct(
    filepath,
    input_dir=None,
    output_dir=None,
    input_ext=".nii.gz",
    plane="axial",
    sample_dicom_fp=os.path.join(os.path.dirname(""), "./sample_dicom.dcm"),
    window_center=40,
    window_width=350,
):
    dataobj = nib.load(filepath)
    headers = dataobj.header
    voxel_arr = dataobj.get_fdata()
    pixdim = headers["pixdim"][1:4].tolist()

    # NIFTI (RAS) -> DICOM (LPI) coordinates
    # i, Left/Right = sagittal plane
    # j, Anterior/Posterior = coronal plane
    # k, Superior/Inferior = axial plane
    voxel_arr = np.flip(voxel_arr, 0)
    voxel_arr = np.flip(voxel_arr, 1)
    voxel_arr = np.flip(voxel_arr, 2)

    # Image coordinates -> World coordinates
    if plane == "axial":
        slice_axis = 2
        plane_axes = [0, 1]
    elif plane == "coronal":
        slice_axis = 1
        plane_axes = [0, 2]
    elif plane == "sagittal":
        slice_axis = 0
        plane_axes = [1, 2]
    thickness = pixdim[slice_axis]
    spacing = [pixdim[plane_axes[1]], pixdim[plane_axes[0]]]
    voxel_arr = np.swapaxes(voxel_arr, *plane_axes)

    # generate DICOM UIDs (StudyInstanceUID and SeriesInstanceUID)
    study_uid = pydicom.uid.generate_uid(prefix=None)
    series_uid = pydicom.uid.generate_uid(prefix=None)

    # randomized patient ID
    patient_id = str(uuid.uuid4())
    patient_name = patient_id

    try:
        scale_slope = str(int(headers["scl_slope"]))
    except ValueError:  # handle NaN
        scale_slope = "1"
    try:
        scale_intercept = str(int(headers["scl_inter"]))
    except ValueError:  # handle NaN
        scale_intercept = "0"

    for slice_index in range(voxel_arr.shape[slice_axis]):
        # generate SOPInstanceUID
        instance_uid = pydicom.uid.generate_uid(prefix=None)

        loc = slice_index * thickness

        ds = pydicom.dcmread(sample_dicom_fp)

        # delete tags
        del ds[0x00200052]  # Frame of Reference UID
        del ds[0x00201040]  # Position Reference Indicator

        # slice and set PixelData tag
        axes = [slice(None)] * 3
        axes[slice_axis] = slice_index
        arr = voxel_arr[tuple(axes)].astype(_get_datatype(headers))
        ds[0x7FE00010].value = arr.tobytes()

        # modify tags
        # - UIDs are created by pydicom.uid.generate_uid at each level above
        # - image position is calculated by combination of slice index and slice thickness
        # - slice location is set to the value of image position along z-axis
        # - Rows/Columns determined by array shape
        # - we set slope/intercept to 1/0 since we're directly converting from PNG pixel values
        ds[0x00080018].value = instance_uid  # SOPInstanceUID
        ds[0x00100010].value = patient_name
        ds[0x00100020].value = patient_id
        ds[0x0020000D].value = study_uid  # StudyInstanceUID
        ds[0x0020000E].value = series_uid  # SeriesInstanceUID
        ds[0x0008103E].value = ""  # Series Description
        ds[0x00200011].value = "1"  # Series Number
        ds[0x00200012].value = str(slice_index + 1)  # Acquisition Number
        ds[0x00200013].value = str(slice_index + 1)  # Instance Number
        ds[0x00201041].value = str(loc)  # Slice Location
        ds[0x00280010].value = arr.shape[0]  # Rows
        ds[0x00280011].value = arr.shape[1]  # Columns
        ds[0x00280030].value = spacing  # Pixel Spacing
        ds[0x00281050].value = str(window_center)  # Window Center
        ds[0x00281051].value = str(window_width)  # Window Width
        ds[0x00281052].value = str(scale_intercept)  # Rescale Intercept
        ds[0x00281053].value = str(scale_slope)  # Rescale Slope

        # Image Position (Patient)
        # Image Orientation (Patient)
        if plane == "axial":
            ds[0x00200032].value = ["0", "0", str(loc)]
            ds[0x00200037].value = ["1", "0", "0", "0", "1", "0"]
        elif plane == "coronal":
            ds[0x00200032].value = ["0", str(loc), "0"]
            ds[0x00200037].value = ["1", "0", "0", "0", "0", "1"]
        elif plane == "sagittal":
            ds[0x00200032].value = [str(loc), "0", "0"]
            ds[0x00200037].value = ["0", "1", "0", "0", "0", "1"]

        # add new tags
        # see tag info e.g., from https://dicom.innolitics.com/ciods/nm-image/nm-reconstruction/00180050
        # Slice Thickness
        ds[0x00180050] = pydicom.dataelem.DataElement(0x00180050, "DS", str(thickness))

        # Output DICOM filepath
        # For root directory of data/, then:
        # e.g., 'data/x/y/z.nii.gz' becomes '{output_dir}/data/x/y/z/{001-999}.dcm'
        dicom_fp = os.path.join(
            output_dir,
            os.path.dirname(filepath).strip("/"),  # remove leading and trailing slashes
            os.path.basename(filepath).replace(input_ext, ""),
            "{:03}.dcm".format(slice_index + 1),
        )

        # create directory
        if not os.path.exists(os.path.dirname(dicom_fp)):
            os.makedirs(os.path.dirname(dicom_fp))

        # write DICOM to file
        pydicom.dcmwrite(dicom_fp, ds)
