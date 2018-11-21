from PIL import Image
import tensorflow as tf
import sys
import pydicom
import numpy as np
import io
import hashlib
from object_detection.utils import dataset_util
from mdai import visualize


def create_tf_bbox_example(annotations, image_id, classes_dict):

    image = visualize.load_dicom_image(image_id)
    width = int(image.shape[1])
    height = int(image.shape[0])

    raw_img = visualize.load_dicom_image(image_id, to_RGB=True)
    img = Image.fromarray(raw_img)
    img_buffer = io.BytesIO()
    img.save(img_buffer, format="jpeg")
    encoded_jpg = Image.open(img_buffer)

    if encoded_jpg.format != "JPEG":
        raise ValueError("Image format not JPEG")

    key = hashlib.sha256(img_buffer.getvalue()).hexdigest()

    xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = []  # List of normalized right x coordinates in bounding box (1 per box)
    ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = []  # List of normalized bottom y coordinates in bounding box (1 per box)
    classes_text = []  # List of string class name of bounding box (1 per box)
    classes = []  # List of integer class id of bounding box (1 per box)

    # per annotation
    for a in annotations:
        w = int(a["data"]["width"])
        h = int(a["data"]["height"])

        x_min = int(a["data"]["x"])
        y_min = int(a["data"]["y"])
        x_max = x_min + w
        y_max = y_min + h

        # WARN: these are normalized
        xmins.append(float(x_min / width))
        xmaxs.append(float(x_max / width))
        ymins.append(float(y_min / height))
        ymaxs.append(float(y_max / height))

        classes_text.append(a["labelId"].encode("utf8"))
        classes.append(classes_dict[a["labelId"]]["class_id"])

    # print(classes)

    tf_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/height": dataset_util.int64_feature(height),
                "image/width": dataset_util.int64_feature(width),
                "image/filename": dataset_util.bytes_feature(image_id.encode("utf8")),
                "image/source_id": dataset_util.bytes_feature(image_id.encode("utf8")),
                "image/key/sha256": dataset_util.bytes_feature(key.encode("utf8")),
                "image/encoded": dataset_util.bytes_feature(img_buffer.getvalue()),
                "image/format": dataset_util.bytes_feature("jpg".encode("utf8")),
                "image/object/bbox/xmin": dataset_util.float_list_feature(xmins),
                "image/object/bbox/xmax": dataset_util.float_list_feature(xmaxs),
                "image/object/bbox/ymin": dataset_util.float_list_feature(ymins),
                "image/object/bbox/ymax": dataset_util.float_list_feature(ymaxs),
                "image/object/class/text": dataset_util.bytes_list_feature(classes_text),
                "image/object/class/label": dataset_util.int64_list_feature(classes),
            }
        )
    )

    return tf_example


def write_to_tfrecords(output_path, dataset):
    """Write images and annotations to tfrecords.
    Args:
        output_path (str): Output file path of the TFRecord.
        dataset (object): Mdai dataset object.
    Examples:

        >>> train_record_fp = os.path.abspath('./train.record')
        >>> export.write_to_tfrecords(train_record_fp, train_dataset, label_ids_dict)
    """

    def _print_progress(count, total):
        # Percentage completion.
        pct_complete = float(count) / total

        # Status-message.
        # Note the \r which means the line should overwrite itself.
        msg = "\r- Progress: {0:.1%}".format(pct_complete)

        # Print it.
        sys.stdout.write(msg)
        sys.stdout.flush()

    print("\nOutput File Path: %s" % output_path)
    writer = tf.python_io.TFRecordWriter(output_path)
    num_images = len(dataset.image_ids)
    for i, image_id in enumerate(dataset.image_ids):
        _print_progress(count=i, total=num_images - 1)
        annotations = dataset.imgs_anns[image_id]
        tf_example = create_tf_bbox_example(annotations, image_id, dataset.classes_dict)
        writer.write(tf_example.SerializeToString())
    writer.close()
