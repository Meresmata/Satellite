import argparse

from tensorflow.keras import applications
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_resnet50(img_h: int, img_w: int, num_classes: int):
    # define our MLP network
    base_model = applications.resnet50.ResNet50(weights=None, include_top=False, input_shape=(img_h, img_w, 3))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(rate=0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    mdl = Model(inputs=base_model.input, outputs=predictions)
    return mdl


def run(train_generator, test_generator, epcs: int, mdl: Model, opt):
    # train the model
    mdl.compile(optimizer=opt, loss='categorical_crossentropy',
                metrics=['accuracy', 'mse', 'mean_absolute_percentage_error'])

    stopper = EarlyStopping(monitor='categorical_crossentropy', patience=min(epcs / 4, 500), mode='min',
                            restore_best_weights=True)

    checker = ModelCheckpoint(monitor='categorical_crossentropy', filepath='weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                              save_best_only=True,
                              save_freq='epoch')

    shower = TensorBoard(histogram_freq=100, update_freq='epoch')
    reducer = ReduceLROnPlateau(factor=0.6, patience=100, min_delta=1e-4, cooldown=100)

    history = model.fit(train_generator, epochs=epcs, verbose=0,
                        validation_data=test_generator,
                        callbacks=[stopper, checker, shower, reducer])

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('train_path', type=str, help='Path to the train main folder of files.')
    parser.add_argument('test_path', type=str, help='Path to the test main folder of files.')
    args = parser.parse_args()

    train_p = args.train_path
    test_p = args.test_path

    img_height, img_width = 214, 214

    model = create_resnet50(img_height, img_width, num_classes=5)
    adam = Adam(lr=0.0001)

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        vertical_flip=True
    )

    validation_datagen = ImageDataGenerator(
        rescale=1.255
    )

    train_gen = train_datagen.flow_from_directory(directory=train_p, target_size=(img_height, img_width), batch_size=64)
    val_gen = validation_datagen.flow_from_directory(directory=test_p, target_size=(img_height, img_width),
                                                     batch_size=64)

    h = run(train_gen, val_gen, 100, model, adam)

    m_name = 'Model_resnet50_epoch{}_score{:3.2f}.hdf5'.format(100,
                                                               min(h.history['val_mean_absolute_percentage_error']))
    model.save(m_name)
