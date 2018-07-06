import random 

from PIL import Image 

import tensorflow as tf 

def hex2rgb(h):
    """Convert Hexcolor encoding to RGB color"""
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2 ,4))

# TODO: check that validation_split is greater than 0 and less than 1 
def train_test_split(image_fps, shuffle=True, validation_split=0.1):
    image_fps_list = list(image_fps)
    if shuffle: 
        sorted(image_fps_list)
        random.seed(42)
        random.shuffle(image_fps_list)

    split_index = int((1 - validation_split) * len(image_fps_list))
    image_fps_train = image_fps_list[:split_index]
    image_fps_val = image_fps_list[split_index:]

    print("Num of instances for training set: %d, validation set: %d" 
          % (len(image_fps_train), len(image_fps_val)))
    return image_fps_train, image_fps_val

def show_statistics():
    pass 

# Timer helper class for benchmarking reading methods
class Timer(object):
    """Timer class
       Wrap a will with a timing function
    """
    
    def __init__(self, name):
        self.name = name
        
    def __enter__(self):
        self.t = time.time()
        
    def __exit__(self, *args, **kwargs):
        print("{} took {} seconds".format(
        self.name, time.time() - self.t))

##################################################################################################
# Consider putting the following TFRecords related functions to export.py or tfrecords.py module 
# Or, create a folder called export and put stuff in there 
###################################################################################################

# OUTPATH = os.path.abspath('./test.record')
# writer = tf.python_io.TFRecordWriter(OUTPATH)

# counter = 0
# len_images = len(image_ids_train)

# for image_id in image_ids_train[:100]:
#     annotations = images_and_annotations[image_id]
#     tf_example = create_tf_example_bbox(annotations, image_id, label_ids)
#     writer.write(tf_example.SerializeToString())

#     if counter % 10 == 0:
#         print("Percent done", (counter/len_images)*100)
#         counter += 1
# writer.close()

def export_json():
    """ Export to json format. """
    pass 

# DICOM 
def create_tf_bbox_example_DICOM(annotations, image_id, label_ids):
    """ TF Example for Bounding Box 
    """
    
    # TODO: figure out how to compress image? 
    #with tf.gfile.GFile(image_id, 'rb') as fid:
    #    encoded_png = fid.read()
    #encoded_png_io = io.BytesIO(encoded_png)
    
    # BUG:  We are using uncompressed image, 
    # should be using compressed png!! 
    ds = pydicom.read_file(image_id)
    image = ds.pixel_array
    
    # If grayscale. Convert to RGB for consistency.
    if len(image.shape) != 3 or image.shape[2] != 3:
        image = np.stack((image,) * 3, -1)

    # BUG: this is WRONG! File is raw, not png 
    image_format = 'png'.encode() 
    
    ### For Bounding Box Annotation Mode 
    image = np.asarray(image)
    width = int(image.shape[1])
    height = int(image.shape[0])
    
    # normalized values 
    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)
    
    # per annotation 
    for a in annotations:
        w = int(a['data']['width'])
        h = int(a['data']['height'])
            
        x_min = int(a['data']['x'])
        y_min = int(a['data']['y'])
        x_max = x_min + w 
        y_max = y_min + h 
        
        # WARN: these are normalized 
        xmins.append(float(x_min/ width))
        xmaxs.append(float(x_max / width))
        ymins.append(float(y_min / height))
        ymaxs.append(float(y_max / height))
      
        classes_text.append(a['labelId'].encode('utf8'))
        classes.append(label_ids[a['labelId']]) 
    
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(image_id.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(image_id.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(image.tostring()), # ?? 
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    
    return tf_example


# dataset is images_and_annotations
# see https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md

def create_tf_bbox_example(annotations, image_id, label_ids):
    
    # WARNI: We are using uncompressed image, 
    # consider using compressed png? Does this make sense for our DICOM images
    # uncompressed images are about 3 MB each 
    ds = pydicom.read_file(image_id)
    image = ds.pixel_array
    
    # TODO: is this necessary?? 
    # If grayscale. Convert to RGB for consistency.
    if len(image.shape) != 3 or image.shape[2] != 3:
        image = np.stack((image,) * 3, -1)   
    
    # For Bounding Box Annotation Mode 
    image = np.asarray(image)

    width = int(image.shape[1])
    height = int(image.shape[0])
    
    #########################################
    # save to file 
    im = Image.fromarray(image)
    
    image_id = image_id+'.jpg'
    im.save(image_id)
    
    with tf.gfile.GFile(image_id, 'rb') as fid:
        encoded_jpg = fid.read()
  
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    
    image = Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    
    key = hashlib.sha256(encoded_jpg).hexdigest()
    ##############################################
        
    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)
    
    # per annotation 
    for a in annotations:
        w = int(a['data']['width'])
        h = int(a['data']['height'])
            
        x_min = int(a['data']['x'])
        y_min = int(a['data']['y'])
        x_max = x_min + w 
        y_max = y_min + h 
        
        # WARN: these are normalized 
        xmins.append(float(x_min/ width))
        xmaxs.append(float(x_max / width))
        ymins.append(float(y_min / height))
        ymaxs.append(float(y_max / height))
      
        classes_text.append(a['labelId'].encode('utf8'))
        classes.append(label_ids[a['labelId']]) 
        
    #print(classes)
    
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(image_id.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(image_id.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes)
    }))
    
    return tf_example

def write_to_tfrecords(output_path, image_ids): 
    """
    Usage: 
        
        TRAIN_OUTPATH = os.path.abspath('./train.record')
        VALID_OUTPATH = os.path.abspath('./valid.record')
        write_to_tfrecords(TRAIN_OUTPATH, image_ids_train)
        write_to_tfrecords(VALID_OUTPATH, image_ids_val)
     
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

    print('\nOutput File Path: %s' % output_path)
    writer = tf.python_io.TFRecordWriter(output_path)
    num_images = len(image_ids)
    for i, image_id in enumerate(image_ids):
        _print_progress(count=i, total=num_images-1)
        annotations = images_and_annotations[image_id]
        tf_example = create_tf_bbox_example(annotations, image_id, label_ids)
        writer.write(tf_example.SerializeToString())
    writer.close()


def tfrecord_decode_jpg(filename_queue): 

    ## BUG: hardcoded!! 
    HEIGHT = 1024
    WIDTH = 1024

    # Construct a Reader to read examples from the .tfrecords file
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'image/height' : tf.FixedLenFeature([], tf.int64),
            'image/width' : tf.FixedLenFeature([], tf.int64),
            'image/filename' : tf.FixedLenFeature([], tf.string),
            'image/encoded' : tf.FixedLenFeature([], tf.string), 
            'image/object/bbox/xmin' : tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymin' : tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/xmax' : tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymax' : tf.VarLenFeature(dtype=tf.float32), 
            'image/object/class/text': tf.VarLenFeature(dtype=tf.string),
            'image/object/class/label': tf.VarLenFeature(dtype=tf.int64)
        }
    )

    height = tf.cast(features['image/height'], tf.int32)
    width = tf.cast(features['image/width'], tf.int32)

    image_shape = tf.stack([HEIGHT, WIDTH, 3])

    image = tf.image.decode_jpeg(features['image/encoded'])
    image_id = features['image/filename']

    xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
    xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
    ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

    #num_bboxes = tf.cast(features['image/object/count'], tf.int32)
    label_text = features['image/object/class/text'].values
    label = features['image/object/class/label'].values

    bboxes = tf.concat(axis=0, values=[xmin, ymin, xmax, ymax])
    bboxes = tf.transpose(bboxes, [1, 0])
    
    return image_id, image, height, width, bboxes, label_text, label

