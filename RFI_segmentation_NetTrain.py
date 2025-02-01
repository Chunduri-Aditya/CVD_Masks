import cv2
import os
import segmentation_models as sm
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import mlflow
import mlflow.tensorflow
from mlflow import MlflowClient
from skimage.io import imshow
from tensorflow.keras.callbacks import TensorBoard
import datetime
import toml


def display(display_list, title=None):
    plt.figure(figsize=(15, 15))
    if title is None:
        title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        if display_list[i].shape[-1] == 1:
            plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]), cmap='gray')
        elif display_list[i].ndim == 2:
            plt.imshow(display_list[i], cmap='gray')
        else:
            plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        # plt.imshow(display_list[i])
        plt.axis('off')
    plt.show()


def create_lookup_table(image):
    # image is an m x n x 3 tensor representing an RGB image
    # Convert the image to float32
    image = tf.cast(image, tf.float32)
    # Convert the RGB tuples to integers
    int_image = tf.cast(tf.tensordot(image, [65536.0, 256.0, 1.0], axes=1), tf.int64)
    # Find the unique integer values in the image
    unique_ints = tf.unique(tf.reshape(int_image, (-1)))[0]
    # Create a lookup table to map the integer values in int_image to their corresponding encoded values
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=unique_ints,
            values=tf.one_hot(tf.range(tf.size(unique_ints)), depth=tf.size(unique_ints)),
            key_dtype=tf.int64,
            value_dtype=tf.float32
        ),
        default_value=tf.zeros(tf.size(unique_ints), dtype=tf.float32)
    )
    return table


def encode_rgb(mask, palette):
    # Create a class map for each color in the palette
    semantic_map = []
    for color in palette:
        class_map = tf.reduce_all(tf.equal(mask, color), axis=-1)
        semantic_map.append(class_map)

    # Stack the class maps to create a one-hot encoded semantic map
    semantic_map = tf.stack(semantic_map, axis=-1)
    semantic_map = tf.cast(semantic_map, tf.float32)
    return semantic_map


def process_path(image_path, vessel_path, av_path):
    img = tf.io.read_file(image_path)
    img = tfio.experimental.image.decode_tiff(img)
    img = img[..., :3]
    img = tf.image.convert_image_dtype(img, tf.float32)

    ves = tf.io.read_file(vessel_path)
    ves = tf.image.decode_png(ves, channels=1)

    av = tf.io.read_file(av_path)
    av = tf.image.decode_png(av, channels=3)

    img = tf.image.resize(img, (512, 512), method='nearest')
    ves = tf.image.resize(ves, (512, 512), method='nearest')
    av = tf.image.resize(av, (512, 512), method='nearest')

    img = img / 255
    ves = ves / 255
    av = av / 255

    img_concat = tf.concat([img, ves], axis=-1)
    return img_concat, av


def process_aug_dataset(image_path, vessel_path, av_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)

    ves = tf.io.read_file(vessel_path)
    ves = tf.image.decode_png(ves, channels=1)

    av = tf.io.read_file(av_path)
    av = tf.image.decode_png(av, channels=3)

    img = tf.image.resize(img, (512, 512), method='nearest')
    ves = tf.image.resize(ves, (512, 512), method='nearest')
    av = tf.image.resize(av, (512, 512), method='nearest')

    img = img / 255
    ves = ves / 255
    av = av / 255

    img_concat = tf.concat([img, ves], axis=-1)
    return img_concat, av


def data_loading_from_files():
    # DATASET_DIR = "D:\\hanx7\\OneDrive - University of Cincinnati\\1_Project\\230829_ML Cardiovascular\\RITE Dataset"
    DATASET_DIR = "../RFI_Dataset/RITE Dataset"

    TRAIN_DATA_DIR = os.path.join(DATASET_DIR, "training", "images")
    TRAIN_VESSEL_DIR = os.path.join(DATASET_DIR, "training", "vessel")
    TRAIN_AV_DIR = os.path.join(DATASET_DIR, "training", "av")
    TEST_DATA_DIR = os.path.join(DATASET_DIR, "test", "images")
    TEST_VESSEL_DIR = os.path.join(DATASET_DIR, "test", "vessel")
    TEST_AV_DIR = os.path.join(DATASET_DIR, "test", "av")

    train_image_list = sorted(os.listdir(TRAIN_DATA_DIR))
    # train_image_list = [x for x in train_image_list if x[-4:] == ".png"]
    train_vessel_list = sorted(os.listdir(TRAIN_VESSEL_DIR))
    train_av_list = sorted(os.listdir(TRAIN_AV_DIR))

    test_image_list = sorted(os.listdir(TEST_DATA_DIR))
    test_vessel_list = sorted(os.listdir(TEST_VESSEL_DIR))
    test_av_list = sorted(os.listdir(TEST_AV_DIR))

    train_image_list = [os.path.join(TRAIN_DATA_DIR, x) for x in train_image_list]
    train_vessel_list = [os.path.join(TRAIN_VESSEL_DIR, x) for x in train_vessel_list]
    train_av_list = [os.path.join(TRAIN_AV_DIR, x) for x in train_av_list]

    test_image_list = [os.path.join(TEST_DATA_DIR, x) for x in test_image_list]
    test_vessel_list = [os.path.join(TEST_VESSEL_DIR, x) for x in test_vessel_list]
    test_av_list = [os.path.join(TEST_AV_DIR, x) for x in test_av_list]

    # Construct the original training & testing datasets
    train_image_filenames = tf.constant(train_image_list)
    train_vessel_filenames = tf.constant(train_vessel_list)
    train_av_filenames = tf.constant(train_av_list)

    test_image_filenames = tf.constant(test_image_list)
    test_vessel_filenames = tf.constant(test_vessel_list)
    test_av_filenames = tf.constant(test_av_list)

    train_dataset_original = tf.data.Dataset.from_tensor_slices(
        (train_image_filenames, train_vessel_filenames, train_av_filenames))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_image_filenames, test_vessel_filenames, test_av_filenames))

    train_dataset_original = train_dataset_original.map(process_path)
    test_dataset = test_dataset.map(process_path)

    # load the training/test dataset (tf.Dataset Class)
    train_dataset, test_dataset = data_loading_from_files()  # TODO: change this to data_loading_from_pickle()

    train_img, train_av = next(train_dataset.take(1).as_numpy_iterator())

    palette_data = np.unique(train_av.reshape(-1, 3), axis=0)  # find out how many colors are there in an RFI image

    # encode the dataset to RGB color
    train_dataset = train_dataset.map(lambda img, av: (img, encode_rgb(av, palette_data)))  # multi-hot to one-hot
    test_dataset = test_dataset.map(lambda img, av: (img, encode_rgb(av, palette_data)))

    return train_dataset, test_dataset


def data_loading_from_pickle():
    # TODO: complete this function

    return


def NetworkTrain_main(data, config, history=None):

    # load the configuration
    EPOCHS = 1 if config['num_epochs'] is None else config['num_epoch']
    BATCH_SIZE = 4 if config['batch_size'] is None else config['batch_size']
    BUFFER_SIZE = 4 if config['buffer_size'] is None else config['buffer_size']

    train_dataset, test_dataset = data[0], data[1]

    train_dataset = train_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    pretrained_base_model = sm.Unet(encoder_weights='imagenet', classes=5)
    inp = keras.Input(shape=(None, None, 4))
    l1 = keras.layers.Conv2D(3, (1, 1))(inp)  # map N channels data to 3 channels
    out = pretrained_base_model(l1)

    pretrained_model = keras.Model(inp, out, name=pretrained_base_model.name)

    # TODO: add useful callbacks: earlyStop, Checkpoint,
    backup_callback = keras.callbacks.BackupAndRestore(backup_dir="./keras_backups")

    # TODO: configure the customized optimizer
    pretrained_model.compile(
        'Adam',
        loss=keras.losses.categorical_crossentropy,
        metrics=[sm.metrics.IOUScore(), sm.metrics.FScore(), keras.metrics.AUC(), keras.metrics.CategoricalAccuracy()])
    # TODO: test the metrics IOUScore vs mIOU, mACC

    history = pretrained_model.fit(train_dataset, epochs=EPOCHS, callbacks=[backup_callback],
                                   validation_data=test_dataset)

    loss = history['loss']
    # TODO: what's in the history? loss, metrics

    return history


if __name__ == "__main__":

    # configure training process
    # TODO: does random seed need to be configured and where to use it?
    config = {
        "folder_name": '../saved_models/V0/',
        "learning_rate": None,
        "num_epochs": 150,
        "batch_size": 4,
        "buffer_size": 4,
        "weight_init": None,
        "overwriting": True,
        "history_path": None
    }

    if not os.path.exists(config['folder_name']):
        os.makedirs(config['folder_name'])
    else:
        if not config['overwriting']:
            raise ValueError("Warning: model version overwriting!")     # avoid overwriting existing version by mistake

    # load data
    data = data_loading_from_files()

    # load history
    # if config['history_path']:

    # training process
    # history = []
    history = NetworkTrain_main(data, config)

    # save data


    # save model
    # model_path =

    # save configuration
    config_path = config['folder_name'] + './model_config.tml'
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            toml.dump(config, f)