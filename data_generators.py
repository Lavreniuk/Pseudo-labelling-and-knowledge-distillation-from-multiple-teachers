import os
import random

import gdal
import keras
import numpy as np
import rasterio
import scipy
from scipy.ndimage import rotate
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def trainGenerator(
    batch_size, train_path, test_path, ker, onehot_encoder, label_encoder
):
    x = []
    y = []
    for i in random.sample(os.listdir(test_path), batch_size):
        rot = np.random.randint(360)
        img = scipy.misc.imread(os.path.join(test_path, i))
        img = rotate(img.astype(np.uint8), rot, reshape=False, order=0)
        y.append(img)
        temp = []
        for j in range(1):
            # img = scipy.misc.imread(os.path.join(train_path,i.replace('mask',str(j))))
            # img = (img-[ 786.,  614.,  427.,  337.])/([ 4405.,  4500.,  4772.,  5488.])
            im_tr = gdal.Open(os.path.join(train_path, i.replace("mask", str(j))))
            im_tr_full = []
            for b in range(im_tr.RasterCount):
                im_tr_full.append(im_tr.GetRasterBand(b + 1).ReadAsArray())
            img = np.array(im_tr_full).transpose(1, 2, 0)
            img = rotate(img, rot, reshape=False, order=0)
            #            img = (img-[ 786.,  614.,  427.,  337.])/([ 4405.,  4500.,  4772.,  5488.])
            temp.append(img)
        x.append(np.concatenate(temp, axis=2))

    x = np.array(x)
    y = np.array(y)
    for i in range(x.shape[0]):
        rot = np.random.randint(4)
        x[i] = np.rot90(x[i], rot)
        y[i] = np.rot90(y[i], rot)

        if np.random.randint(2):
            flip = np.random.randint(2)
            x[i] = np.flip(x[i], flip)
            y[i] = np.flip(y[i], flip)

    img = x
    mask = onehot_encoder.transform(
        label_encoder.transform(
            y.reshape(
                -1,
            )
        ).reshape(-1, 1)
    ).reshape((batch_size, 2 * ker, 2 * ker, -1))
    return img, mask


class DataGen(keras.utils.Sequence):
    def __init__(
        self,
        ids,
        path_images,
        path_labels,
        batch_size=8,
        image_size=224,
        n_classes=2,
        type="train",
    ):
        self.ids = ids
        self.path_images = path_images
        self.path_labels = path_labels
        self.batch_size = batch_size
        self.image_size = image_size
        self.on_epoch_end()
        self.gt_names = sorted(os.listdir(path_labels))
        self.label_encoder = LabelEncoder()
        self.train_labels_encoded = self.label_encoder.fit_transform(
            np.arange(n_classes + 1)
        )
        self.onehot_encoder = OneHotEncoder(sparse=False).fit(
            self.train_labels_encoded.reshape(-1, 1)
        )
        self.n_classes = n_classes
        self.threshold_aug = 0.5

    def __load__(self, id_name):
        img_name = self.gt_names[int(id_name)]
        image_path = os.path.join(
            self.path_images, img_name[: img_name.find("_")] + "_0.tif"
        )
        mask_path = os.path.join(self.path_labels, img_name)
        image = rasterio.open(image_path).read().astype(np.float32)
        mask = rasterio.open(mask_path).read().astype(np.float32)
        mask = self.onehot_encoder.transform(
            self.label_encoder.transform(
                mask.reshape(
                    -1,
                )
            ).reshape(-1, 1)
        ).reshape((self.image_size, self.image_size, -1))

        return image, mask

    def __getitem__(self, index):
        if (index + 1) * self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index * self.batch_size
        files_batch = self.ids[index * self.batch_size : (index + 1) * self.batch_size]

        image = []
        mask = []
        for id_name in files_batch:
            _img, _mask = self.__load__(id_name)
            if type == "train":
                _img, _mask = self.augmentation(_img, _mask)
            image.append(_img)
            mask.append(_mask)

        image = np.array(image)
        image = np.moveaxis(image, [0, 1, 2, 3], [0, -1, 1, 2])
        mask = np.array(mask)

        return image, mask

    def on_epoch_end(self):
        # self.indexes = np.arange(len(self.list_IDs))
        # if self.shuffle == True:
        #    np.random.shuffle(self.indexes)
        pass

    def augmentation(self, img, mask):
        if np.random.uniform(0, 1) > self.threshold_aug:
            rot = np.random.randint(360)
            img = rotate(img, rot, reshape=False, order=0)
            mask = rotate(mask, rot, reshape=False, order=0)
        if np.random.uniform(0, 1) > self.threshold_aug:
            flip = np.random.randint(2)
            img = np.flip(img, flip)
            mask = np.flip(mask, flip)
        return img, mask

    def __len__(self):
        return int(np.ceil(len(self.ids) / float(self.batch_size)))
