import argparse
import inspect
import math
import os
import time
import warnings

import keras
import numpy as np
import segmentation_models as sm
import segmentation_models_pytorch as smp
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Activation, BatchNormalization, Dense, Dropout, Input
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from data_generators import DataGen, trainGenerator
from loss import (binary_focal_loss, build_masked_loss, categorical_focal_loss,
                  dice_loss, get_loss, jaccard_loss, masked_accuracy,
                  masked_categorical_crossentropy)
from model import get_unet
from utils import map_creating, read_dataset

warnings.simplefilter("ignore", RuntimeWarning)


def training():
    time0 = time.time()
    batch_size = 24  # 10
    save_step = 100
    num_read = len(os.listdir(os.path.join(tif_path, "label_data_tr/")))
    num_read_val = len(os.listdir(os.path.join(tif_path, "label_data_val/")))
    num_steps = 3
    model_loss = "categorical_crossentropy"  #'Jaccard'
    learning_rate_initial = 1e-3
    lr_decreasing_factor = 0.3
    lr_patience = 2
    model_loss = "Jaccard"
    # img, mask = trainGenerator(batch_size,os.path.join(tif_path,'train_data/'),os.path.join(tif_path,'label_data_tr/'),ker,onehot_encoder, label_encoder)
    ids = np.array(
        [i for i in range(len(os.listdir(os.path.join(tif_path, "label_data_tr"))))]
    )
    gen = DataGen(
        ids,
        os.path.join(tif_path, "train_data/"),
        os.path.join(tif_path, "label_data_tr/"),
        batch_size=batch_size,
        image_size=2 * ker,
    )
    ids = np.array(
        [i for i in range(len(os.listdir(os.path.join(tif_path, "label_data_val"))))]
    )
    gen_val = DataGen(
        ids,
        os.path.join(tif_path, "train_data/"),
        os.path.join(tif_path, "label_data_val/"),
        batch_size=batch_size,
        image_size=2 * ker,
        type="val",
    )
    img, mask = gen.__getitem__(1)
    input_img = Input((2 * ker, 2 * ker, img.shape[-1]), name="img")
    if os.path.exists(train_model):
        model = load_model(
            train_model,
            custom_objects={
                "binary_crossentropy_plus_jaccard_loss": sm.losses.bce_jaccard_loss,
                "masked_categorical_crossentropy": masked_categorical_crossentropy,
                "masked_accuracy": masked_accuracy,
            },
            compile=False,
        )
    else:
        model = get_unet(
            input_img, mask.shape[-1], n_filters=16, dropout=0.05, batchnorm=True
        )
        #            model = sm.Unet('efficientnetb3', input_shape=(None, None, img.shape[-1]),classes=3, encoder_weights=None, activation='softmax')
        #            model = sm.FPN('efficientnetb3', input_shape=(None, None, img.shape[-1]),classes=3, encoder_weights=None, activation='softmax')
        #           model = smp.UnetPlusPlus('efficientnet-b3', in_channels = img.shape[-1],classes=3, encoder_weights=None, activation='softmax')

        #            model.compile(optimizer=Adam(), loss=build_masked_loss(K.binary_crossentropy), metrics=[masked_accuracy])
        masked_categorical_crossentropy = get_loss(onehot_encoder.transform([[0]]))
        # model.compile(loss=masked_categorical_crossentropy, optimizer='adam', metrics=[masked_accuracy])
        opt = keras.optimizers.Adam(lr=learning_rate_initial)
        if model_loss == "FL":
            model.compile(
                loss=[categorical_focal_loss(alpha=[[0.5] * classes_num])],
                optimizer=opt,
                metrics=[masked_accuracy],
            )
        elif model_loss == "Dice":
            model.compile(loss=[dice_loss], optimizer=opt, metrics=[masked_accuracy])
        elif model_loss == "Jaccard":
            model.compile(loss=[jaccard_loss], optimizer=opt, metrics=[masked_accuracy])
        else:
            #                model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=[masked_accuracy, sm.metrics.iou_score])
            model.compile(
                loss=sm.losses.bce_jaccard_loss,
                optimizer=opt,
                metrics=[masked_accuracy, sm.metrics.iou_score],
            )

    #            model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
    # print(model.summary())
    plot_model(
        model,
        to_file="model_architechture.png",
        show_shapes=True,
        show_layer_names=True,
        dpi=300,
    )
    print("#" * 100)
    print("Number of layers in the model = ", len(model.layers))
    print("#" * 100)

    callbacks = [
        ReduceLROnPlateau(
            factor=lr_decreasing_factor,
            patience=lr_patience,
            min_lr=0.00001,
            verbose=1,
            monitor="val_masked_accuracy",
            mode="max",
        ),
        ModelCheckpoint(
            f"model-unet_{model_loss}.h5",
            verbose=0,
            save_best_only=True,
            monitor="val_masked_accuracy",
            mode="max",
        ),
    ]
    time0 = time.time()
    model.fit_generator(
        gen,
        steps_per_epoch=int(num_read / batch_size),
        validation_data=gen_val,
        validation_steps=num_read_val,
        epochs=num_steps,
        callbacks=callbacks,
        verbose=1,
    )

    """
    for step in range(num_steps):
        K.set_value(model.optimizer.lr, 0.001/math.pow(2,step))
        #model.optimizer.lr.set_value()
        for z in range(int(num_read/batch_size)):
            #K.set_value(model.optimizer.lr, 0.001/math.pow(10,z*1e-4))
            results = model.fit(img, mask, callbacks=callbacks, verbose=0)
            img, mask = trainGenerator(batch_size,os.path.join(tif_path,'train_data/'),os.path.join(tif_path,'label_data_tr/'),ker,onehot_encoder, label_encoder)

            print("Epoch: {}/{}, Step: {}/{}, Time: {} sec".format(z+1, int(num_read/batch_size), step+1, num_steps, round(time.time()-time0)), "Training loss: {}".format(results.history))
            if z % save_step == 0 or z == int(num_read/batch_size)-1:
#                if results.history['loss'][0]<loss:
                model.save('unet_model_' + str(step+1) + '.h5')
    """
    print("\nOptimization has been Finished!")
    print(round(time.time() - time0))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", nargs="?", default="", help="folder with images path")
    parser.add_argument("-s", nargs="?", default=0, help="shut down machine")
    parser.add_argument("-r", nargs="?", default=0, help="read")
    parser.add_argument("-l", nargs="?", default="", help="load model")
    parser.add_argument("-t", nargs="?", default="", help="train model")

    args = parser.parse_args()

    s = int(args.s)
    map_path = tif_path = os.path.normpath(args.f)
    if tif_path == "":
        tif_path = os.path.dirname(
            os.path.abspath(inspect.getframeinfo(inspect.currentframe()).filename)
        )

    ker = 32
    step = 32
    read = int(args.r)
    train_model = args.t
    load_my_model = args.l

    label_encoder = LabelEncoder()
    classes_num = len(np.arange(3))
    train_labels_encoded = label_encoder.fit_transform(np.arange(3))
    onehot_encoder = OneHotEncoder(sparse=False).fit(
        train_labels_encoded.reshape(-1, 1)
    )

    if read:
        read_dataset(tif_path, ker, step)

    if train_model != "":
        training()

    if map_path != "":
        map_creating(load_my_model, map_path, tif_path, ker, label_encoder)
