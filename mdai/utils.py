import random 

import tensorflow as tf 




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


def export_json():
    """ Export to json format. """
    pass 



def export_tfrecords():
    """ Export to TensorFlow's TFRecord format. """
    print('Export to TFRecords')


#####################
# test tf export 
#####################

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

def create_tf_example_bbox(annotations, image_id, label_ids):
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

