import argparse
import io
import itertools
import os
import typing as tp

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from six.moves import range
from sklearn.metrics import confusion_matrix
from tensorflow.keras import applications
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, DirectoryIterator

val_gen: DirectoryIterator
mdl: Model


def log_confusion_matrix(epoch, logs):
    """
    add a confusion matrix for the image tab in tensorboard
    :param epoch: int, current epoch number
    :param logs: tp.List, the metrics and loss of the current epoch
    :return: None
    """
    # Use the model to predict the values from the validation dataset.
    # create list of 128 images, labels
    global val_gen
    global mdl
    itx = 128 // bch_size
    test_images, test_labels_raw = [], []

    for i in range(itx):
        tmp_img, tmp_lbs = next(val_gen)
        test_images.extend(tmp_img)
        test_labels_raw.extend(tmp_lbs)

    class_dict = {v: k for k, v in val_gen.class_indices.items()}

    test_pred_raw = mdl.predict(np.array(test_images))
    test_pred = np.argmax(test_pred_raw, axis=1)
    test_pred = np.array([class_dict[x] for x in test_pred])
    test_labels = np.argmax(test_labels_raw, axis=1)
    test_labels = np.array([class_dict[x] for x in test_labels])

    # Calculate the confusion matrix.
    cm = confusion_matrix(test_labels, test_pred)
    # Log the confusion matrix as an image summary.
    figure = plot_confusion_matrix(cm, class_names=[x for x in val_gen.class_indices.keys()])
    cm_image = plot_to_image(figure)

    # Log the confusion matrix as an image summary.
    with file_writer_cm.as_default():
        tf.summary.image("Confusion Matrix", cm_image, step=epoch,
                         description='with Validation loss: {}'.format(logs['val_loss']))


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title("Confusion matrix")
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max(initial=0) / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def plot_to_image(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def create_net(base_net: tp.Callable, img_h: int, img_w: int, num_classes: int):
    """
    create a net based upon a base net, add the top
    :param base_net: tp.Callable
    :param img_h: int
    :param img_w: int
    :param num_classes: int
    :return: Model
    """
    # define our MLP network
    global mdl
    base_model = base_net(weights="imagenet", include_top=False, input_shape=(img_h, img_w, 3))

    # add the top of the network, to get the correct number of classes as output
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(rate=0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    mdl = Model(inputs=base_model.input, outputs=predictions)
    return mdl


def run(train_path: str, test_path: str, save_dir: str, epcs: int, batch_size: int, _mdl: Model,
        opt: tf.keras.optimizers):
    """
    train a single model, the model and the validation generator are used as global variable, to use both in the lambda callback,
    is there a better idea?
    :param train_path: str
    :param test_path: str
    :param save_dir: str
    :param epcs: int
    :param batch_size: int
    :param opt: tf.keras.optimizers
    :param _mdl: Model
    :return: history
    """
    # train the model
    path = save_dir + '/weights.{epoch:03d}-{val_accuracy:.3f}.hdf5'
    checker = ModelCheckpoint(monitor='val_loss', filepath=path,
                              save_best_only=True, save_freq='epoch')
    shower = TensorBoard(histogram_freq=1)
    cm_callback = LambdaCallback(on_epoch_end=log_confusion_matrix)

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=(0.8, 1.3),
        channel_shift_range=0.5
    )
    train_gen = train_datagen.flow_from_directory(directory=train_path, target_size=(img_height, img_width),
                                                  batch_size=batch_size)

    validation_datagen = ImageDataGenerator(
        rescale=1. / 255
    )
    global val_gen
    val_gen = validation_datagen.flow_from_directory(directory=test_path, target_size=(img_height, img_width),
                                                     batch_size=batch_size)

    global mdl
    mdl = _mdl

    _loss = 'categorical_crossentropy' if train_gen.num_classes > 2 else 'binary_crossentropy'
    mdl.compile(optimizer=opt, loss=_loss, metrics=['accuracy'])

    history = mdl.fit(train_gen, epochs=epcs, validation_data=val_gen,
                      callbacks=[checker, shower, cm_callback]
                      )

    return history


def multi_eval(test_path: str, model_parent_path: str):
    """
    evaluate all models, that have the parent folder "model_parent_path" with the the test image in  "test_path"
    :param test_path: str
    :param model_parent_path: str
    :return: None
    """
    datagen = ImageDataGenerator(
        rescale=1. / 255
    )
    gen = datagen.flow_from_directory(directory=test_path, target_size=(img_height, img_width))

    for _folder, _, _files in os.walk(model_parent_path):
        _model_files = [file for file in _files if file.endswith("hdf5")]
        for model_file in _model_files:
            _model = load_model(os.path.join(_folder, model_file))

            test_loss = _model.evaluate(gen, verbose=0)
            print("{}, {}: {}".format(_folder, model_file, test_loss[1]))


def multi_run(train_path: str, test_path: str, save_dirs: tp.List[str], epcs: int, batch_size: int,
              mdls: tp.List[Model],
              opt: tf.keras.optimizers):
    """
    train several networks on the same dataset
    :param train_path: str
    :param test_path: str
    :param save_dirs: tp.List[str]
    :param epcs: int
    :param batch_size: int
    :param mdls: tp.List[Model]
    :param opt: tf.keras.optimizers
    :return: tp.List[History]
    """
    histories = []
    for save_dir, _mdl in zip(save_dirs, mdls):
        histories.append(run(train_path, test_path, save_dir, epcs, batch_size, _mdl, opt))

    return histories


def get_classifier(images: tp.List[Image.Image], _model_path: str, class_dict: tp.Dict) -> np.ndarray:
    """
    get in classifier for every image in the list
    :param images: tp.List[Image.Image]
    :param _model_path: str
    :param class_dict: tp.Dict
    :return: np.ndarray
    """
    # noinspection PyTypeChecker
    images = [np.asarray(x, dtype="float32") / 255 for x in images]
    images = np.stack(images)

    _model = load_model(_model_path)
    test_preds_raw = _model.predict(images, verbose=0)

    test_preds = np.argmax(test_preds_raw, axis=1)

    #  convert (als mapping) prediction from number to str
    return np.vectorize(lambda x: class_dict[x])(test_preds)


def get_multi_classifier(_images: tp.List[Image.Image], _model_paths: tp.List[str], class_dict: tp.Dict) -> np.ndarray:
    """
    Return the classifiers for every image in the list, tested with different neural networks
    :param _images: tp.List[Image.Image]
    :param _model_paths: tp.List[str]
    :param class_dict: tp.Dict
    :return: tp.List[str]
    """
    classifiers = [get_classifier(_images, x, class_dict) for x in _model_paths]
    return np.array(classifiers).transpose()


def single_classifier(_classifier: np.ndarray) -> str:
    """
    compare the classifiers per image and set them to mixed if different, or the single common value
    :param _classifier: np.ndarray
    :return: str
    """
    if all([x == y for x in _classifier for y in _classifier]):
        return _classifier[0]
    else:
        return "mixed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('train_path', type=str, help='Path to the train main folder of files.')
    parser.add_argument('test_path', type=str, help='Path to the test main folder of files.')
    parser.add_argument('-m', '--model_path', type=str, help='path to model.', default=None)
    args = parser.parse_args()

    train_p = args.train_path
    test_p = args.test_path
    model_path = args.model_path

    img_height, img_width = 214, 214
    epochs, bch_size = 10, 16

    file_writer_cm = tf.summary.create_file_writer('logs/cm')

    save_paths = ["./mobilnetV2",
                  "./ResNet50_V2",
                  "./DenseNet201"]

    nets = [applications.mobilenet_v2.MobileNetV2,
            applications.resnet_v2.ResNet50V2,
            applications.DenseNet201]

    for net, save_path in zip(nets, save_paths):

        for learning_rate in [1e-4]:
            for decay in [1e-6]:
                print(save_path, learning_rate, decay)
                adam = Adam(lr=learning_rate, decay=decay)
                model = create_net(net, img_height, img_width,
                                   num_classes=len(os.listdir(train_p)))

                # rename save_path to include learning rate and decay of the optimizer
                save_path_new = "{}/lr{}_dc{}".format(save_path, learning_rate, decay)
                if not os.path.exists(save_path_new):
                    os.makedirs(save_path_new)

                h = run(train_p, test_p, save_path_new, epochs, bch_size, model, adam)

                model.save(save_path_new + "/final.hdf5")
