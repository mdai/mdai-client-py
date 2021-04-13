import os
import cv2
import numpy as np
import dicom2nifti

DEFAULT_WINDOW_WIDTH = 350
DEFAULT_WINDOW_CENTER = 50
DEFAULT_IMAGE_SIZE = (512.0, 512.0)


def _sort_dicoms(dicoms):
    """
    Sort the dicoms based on the image position patient
    @dicoms: list of dicoms
    """
    # find most significant axis to use during sorting
    # the original way of sorting (first x than y than z) does not work in certain border situations
    # where for exampe the X will only slightly change causing the values to remain equal on multiple slices
    # messing up the sorting completely)
    dicom_input_sorted_x = sorted(dicoms, key=lambda x: (x.ImagePositionPatient[0]))
    dicom_input_sorted_y = sorted(dicoms, key=lambda x: (x.ImagePositionPatient[1]))
    dicom_input_sorted_z = sorted(dicoms, key=lambda x: (x.ImagePositionPatient[2]))
    diff_x = abs(
        dicom_input_sorted_x[-1].ImagePositionPatient[0]
        - dicom_input_sorted_x[0].ImagePositionPatient[0]
    )
    diff_y = abs(
        dicom_input_sorted_y[-1].ImagePositionPatient[1]
        - dicom_input_sorted_y[0].ImagePositionPatient[1]
    )
    diff_z = abs(
        dicom_input_sorted_z[-1].ImagePositionPatient[2]
        - dicom_input_sorted_z[0].ImagePositionPatient[2]
    )
    if diff_x >= diff_y and diff_x >= diff_z:
        return dicom_input_sorted_x
    if diff_y >= diff_x and diff_y >= diff_z:
        return dicom_input_sorted_y
    if diff_z >= diff_x and diff_z >= diff_y:
        return dicom_input_sorted_z


def _apply_slope_intercept(dicom_file):
    """
    Applies rescale slope and rescale intercept transformation.
    """
    array = dicom_file.pixel_array.copy()

    scale_slope = 1
    scale_intercept = 0
    if "RescaleIntercept" in dicom_file:
        scale_intercept = int(dicom_file.RescaleIntercept)
    if "RescaleIntercept" in dicom_file:
        scale_slope = int(dicom_file.RescaleSlope)
    array *= scale_slope
    array += scale_intercept
    return array


def _remove_padding(array):
    """
    Removes background/padding from an 8bit numpy array.
    """
    arr = array.copy()
    nonzeros = np.nonzero(arr)
    x1 = np.min(nonzeros[0])
    x2 = np.max(nonzeros[0])
    y1 = np.min(nonzeros[1])
    y2 = np.max(nonzeros[1])
    return arr[x1:x2, y1:y2]


def _get_window_from_dicom(dicom_file):
    """
    Returns window width and window center values.
    Defaults: width - 350, level - 50.
    """
    if "WindowWidth" in dicom_file:
        width = float(dicom_file.WindowWidth)
    else:
        width = DEFAULT_WINDOW_WIDTH

    if "WindowCenter" in dicom_file:
        level = float(dicom_file.WindowCenter)
    else:
        level = DEFAULT_WINDOW_CENTER
    return width, level


def window(array, width, level):
    """
    Applies windowing operation.
    """
    array = np.clip(array, level - width // 2, level + width // 2)
    return array


def rescale_to_8bit(array):
    """
    Convert an array to 8bit (0-255).
    """
    array = array - np.min(array)
    array = array / np.max(array)
    array = (array * 255).astype("uint8")
    return array


def load_dicom_array(dicom_file, apply_slope_intercept=True):
    """
    Returns the dicom image as a Numpy array.
    """
    array = dicom_file.pixel_array.copy()
    if apply_slope_intercept:
        array = _apply_slope_intercept(dicom_file)
    return array


def convert_dicom_to_nifti(dicom_files, tempdir):
    """
    Converts a dicom series to nifti format.
    Saves nifti in directory provided with filename as SeriesInstanceUID.nii.gz
    Returns a sorted list of dicom files based on image position patient.
    """
    output_file = os.path.join(tempdir, dicom_files[0].SeriesInstanceUID + ".nii.gz")
    nifti_file = dicom2nifti.convert_dicom.dicom_array_to_nifti(
        dicom_files, output_file=output_file, reorient_nifti=True,
    )
    return _sort_dicoms(dicom_files)


def convert_dicom_to_8bit(dicom_file, imsize=None, width=None, level=None, remove_padding=False):
    """
    Given a DICOM file, window specifications, and image size,
    return the image as a Numpy array scaled to [0,255] of the specified size.
    """
    if width is None or level is None:
        width, level = _get_window_from_dicom(dicom_file)

    array = _apply_slope_intercept(dicom_file)
    array = window(array, width, level)
    array = rescale_to_8bit(array)

    if remove_padding:
        array = _remove_padding(array)

    if imsize is not None:
        array = cv2.resize(array, imsize)
    return array


def convert_to_RGB(array, imsize=None):
    """
    Converts a single channel monochrome image to a 3 channel RGB image.
    """
    img = np.stack((array,) * 3, axis=-1)
    if imsize is not None:
        img = cv2.resize(img, imsize)
    return img


def convert_to_RGB_window(array, width, level, imsize=None):
    """
    Converts a monochrome image to 3 channel RGB with windowing.
    Width and level can be lists for different values per channel.
    """
    if type(width) is list and type(level) is list:
        R = window(array, width[0], level[0])
        G = window(array, width[1], level[1])
        B = window(array, width[2], level[2])
        img = np.stack([R, G, B], axis=-1)
    else:
        R = window(array, width, level)
        img = np.stack((R,) * 3, axis=-1)

    if imsize is not None:
        img = cv2.resize(img, imsize)
    return img


def stack_slices(dicom_files):
    """
    Stacks the +-1 slice to each slice in a dicom series.
    Returns the list of stacked images and sorted list of dicom files.
    """
    dicom_files = _sort_dicoms(dicom_files)
    dicom_images = [load_dicom_array(i) for i in dicom_files]

    stacked_images = []
    for i, file in enumerate(dicom_images):
        if i == 0:
            img = np.stack([dicom_images[i], dicom_images[i], dicom_images[i + 1]], axis=-1)
            stacked_images.append(img)
        elif i == len(dicom_files) - 1:
            img = np.stack([dicom_images[i - 1], dicom_images[i], dicom_images[i]], axis=-1)
            stacked_images.append(img)
        else:
            img = np.stack([dicom_images[i - 1], dicom_images[i], dicom_images[i + 1]], axis=-1)
            stacked_images.append(img)

    return stacked_images, dicom_files
