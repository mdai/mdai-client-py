import random 

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


def export_tfrecord():
    """ Export to TensorFlow's TFRecord format. """
    pass 


def export_json():
    """ Export to json format. """
    pass 