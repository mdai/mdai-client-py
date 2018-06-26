import random 

def train_test_split(image_fps, validation_split=0.1):
    image_fps_list = list(image_fps)
    sorted(image_fps_list)
    random.seed(42)
    random.shuffle(image_fps_list)

    split_index = int((1 - validation_split) * len(image_fps_list))

    image_fps_train = image_fps_list[:split_index]
    image_fps_val = image_fps_list[split_index:]

    return image_fps_train, image_fps_val