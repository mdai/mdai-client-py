import pydicom
import numpy as np
import colorsys
import random
import cv2
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches


def random_colors(N, bright=True):
    """ Generate random colors.
    To get visually distinct colors, generate them in HSV space then convert to RGB.

    Args:
        N (int):
            Number of colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


# based on functions in: https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
def display_images(image_ids, titles=None, cols=3, cmap="gray", norm=None, interpolation=None):
    """Display images given image ids.

    Args:
        image_ids (list):
            List of image ids.

    TODO: figsize should not be hardcoded
    """
    titles = titles if titles is not None else [""] * len(image_ids)
    rows = len(image_ids) // cols + 1
    plt.figure(figsize=(14, 14 * rows // cols))
    i = 1
    for image_id, title in zip(image_ids, titles):
        plt.subplot(rows, cols, i)
        plt.title(title, fontsize=9)
        plt.axis("off")

        image = load_dicom_image(image_id, rescale=True)
        plt.imshow(image, cmap=cmap, norm=norm, interpolation=interpolation)

        i += 1
    plt.show()


def load_dicom_image(image_id, to_RGB=False, rescale=False):
    """ Load a DICOM image.

    Args:
        image_id (str):
            image id (filepath).
        to_RGB (bool, optional):
            Convert grayscale image to RGB.

    Returns:
        image array.
    """
    ds = pydicom.dcmread(image_id)
    try:
        image = ds.pixel_array
    except Exception:
        msg = (
            "Could not read pixel array from DICOM with TransferSyntaxUID "
            + ds.file_meta.TransferSyntaxUID
            + ". Likely unsupported compression format."
        )
        print(msg)

    if rescale:
        max_pixel_value = np.amax(image)
        min_pixel_value = np.amin(image)

        if max_pixel_value >= 255:
            # print("Input image pixel range exceeds 255, rescaling for visualization.")
            pixel_range = np.abs(max_pixel_value - min_pixel_value)
            pixel_range = pixel_range if pixel_range != 0 else 1
            image = image.astype(np.float32) / pixel_range * 255
            image = image.astype(np.uint8)

    if to_RGB:
        # If grayscale. Convert to RGB for consistency.
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)

    return image


def load_mask(image_id, dataset):
    """Load instance masks for the given image. Masks can be different types,
    mask is a binary true/false map of the same size as the image.

    """
    # annotations = imgs_anns[image_id]
    annotations = dataset.get_annotations_by_image_id(image_id)
    count = len(annotations)
    print("Number of annotations: %d" % count)

    image = load_dicom_image(image_id)
    width = image.shape[1]
    height = image.shape[0]

    if count == 0:
        print("No annotations")
        mask = np.zeros((height, width, 1), dtype=np.uint8)
        class_ids = np.zeros((1,), dtype=np.int32)
    else:
        mask = np.zeros((height, width, count), dtype=np.uint8)
        class_ids = np.zeros((count,), dtype=np.int32)

        for i, a in enumerate(annotations):

            label_id = a["labelId"]
            annotation_mode = dataset.label_id_to_class_annotation_mode(label_id)
            # print(annotation_mode)

            if annotation_mode == "bbox":
                # Bounding Box
                x = int(a["data"]["x"])
                y = int(a["data"]["y"])
                w = int(a["data"]["width"])
                h = int(a["data"]["height"])
                mask_instance = mask[:, :, i].copy()
                cv2.rectangle(mask_instance, (x, y), (x + w, y + h), 255, -1)
                mask[:, :, i] = mask_instance

            # FreeForm or Polygon
            elif annotation_mode == "freeform" or annotation_mode == "polygon":
                vertices = np.array(a["data"]["vertices"])
                vertices = vertices.reshape((-1, 2))
                mask_instance = mask[:, :, i].copy()
                cv2.fillPoly(mask_instance, np.int32([vertices]), (255, 255, 255))
                mask[:, :, i] = mask_instance

            # Line
            elif annotation_mode == "line":
                vertices = np.array(a["data"]["vertices"])
                vertices = vertices.reshape((-1, 2))
                mask_instance = mask[:, :, i].copy()
                cv2.polylines(mask_instance, np.int32([vertices]), False, (255, 255, 255), 12)
                mask[:, :, i] = mask_instance

            elif annotation_mode == "location":
                # Bounding Box
                x = int(a["data"]["x"])
                y = int(a["data"]["y"])
                mask_instance = mask[:, :, i].copy()
                cv2.circle(mask_instance, (x, y), 7, (255, 255, 255), -1)
                mask[:, :, i] = mask_instance

            elif annotation_mode == "mask":
                mask_instance = mask[:, :, i].copy()
                if a.data["foreground"]:
                    for i in a.data["foreground"]:
                        mask_instance = cv2.fillPoly(mask_instance, [np.array(i, dtype=np.int32)], (255, 255, 255))
                if a.data["background"]:
                    for i in a.data["background"]:
                        mask_instance = cv2.fillPoly(mask_instance, [np.array(i, dtype=np.int32)], (0,0,0))
                mask[:, :, i] = mask_instance

            elif annotation_mode is None:
                print("Not a local instance")

            # load class id
            class_ids[i] = dataset.label_id_to_class_id(label_id)

    return mask.astype(np.bool), class_ids.astype(np.int32)


def apply_mask(image, mask, color, alpha=0.3):
    """Apply the given mask to the image.

    Args:
        image: height, widht, channel.

    Returns:
        image with applied color mask.
    """
    for c in range(3):
        image[:, :, c] = np.where(
            mask == 1, image[:, :, c] * (1 - alpha) + alpha * color[c] * 255, image[:, :, c]
        )
    return image


def extract_bboxes(mask):
    """Compute bounding boxes from masks.

    Args:
        mask [height, width, num_instances]:
            Mask pixels are either 1 or 0.

    Returns:
        bounding box array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)


def get_image_ground_truth(image_id, dataset):
    """Load and return ground truth data for an image (image, mask, bounding boxes).

    Args:
        image_id:
            Image id.

    Returns:
        image:
            [height, width, 3]
        class_ids:
            [instance_count] Integer class IDs
        bbox:
            [instance_count, (y1, x1, y2, x2)]
        mask:
            [height, width, instance_count]. The height and width are those of the image unless
            use_mini_mask is True, in which case they are defined in MINI_MASK_SHAPE.
    """
    # image = load_dicom_image(image_id, to_RGB=True)
    image = load_dicom_image(image_id, to_RGB=True, rescale=True)

    mask, class_ids = load_mask(image_id, dataset)

    _idx = np.sum(mask, axis=(0, 1)) > 0
    mask = mask[:, :, _idx]
    class_ids = class_ids[_idx]

    # Bounding boxes. Note that some boxes might be all zeros
    # if the corresponding mask got cropped out.
    # bbox: [num_instances, (y1, x1, y2, x2)]
    bbox = extract_bboxes(mask)

    return image, class_ids, bbox, mask


def display_annotations(
    image,
    boxes,
    masks,
    class_ids,
    scores=None,
    title="",
    figsize=(16, 16),
    ax=None,
    show_mask=True,
    show_bbox=True,
    colors=None,
    captions=None,
):
    """Display annotations for image.

    Args:
        boxes:
            [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
        masks:
            [height, width, num_instances]
        class_ids:
            [num_instances]
        scores:
            (optional) confidence scores for each box
        title:
            (optional) Figure title
        show_mask, show_bbox:
            To show masks and bounding boxes or not
        figsize:
            (optional) the size of the image
        colors:
            (optional) An array or colors to use with each object
        captions:
            (optional) A list of strings to use as captions for each object
    """

    # Number of instancesload_mask
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis("off")
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                alpha=0.7,
                linestyle="dashed",
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None

            label = class_id
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption, color="w", size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = patches.Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    # ax.imshow(masked_image)
    if auto_show:
        plt.show()


def draw_box_on_image(image, boxes, h, w):
    """Draw box on an image.

    Args:
        image:
            three channel (e.g. RGB) image.
        boxes:
            normalized box coordinate (between 0.0 and 1.0).
        h:
            image height
        w:
            image width
    """

    for i in range(len(boxes)):
        (left, right, top, bottom) = (
            boxes[i][0] * w,
            boxes[i][2] * w,
            boxes[i][1] * h,
            boxes[i][3] * h,
        )
        p1 = (int(left), int(top))
        p2 = (int(right), int(bottom))
        cv2.rectangle(image, p1, p2, (77, 255, 9), 3, 1)
