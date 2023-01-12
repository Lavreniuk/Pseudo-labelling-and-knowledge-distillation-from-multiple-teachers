import os
import sys
import time

import gdal
import numpy as np
import osr
import segmentation_models as sm
from keras.models import load_model


def read_dataset(tif_path, ker, step):
    time0 = time.time()
    if not os.path.exists(os.path.join(tif_path, "train_data/")):
        os.mkdir(os.path.join(tif_path, "train_data/"))
    if not os.path.exists(os.path.join(tif_path, "label_data_tr/")):
        os.mkdir(os.path.join(tif_path, "label_data_tr/"))
    if not os.path.exists(os.path.join(tif_path, "label_data_val/")):
        os.mkdir(os.path.join(tif_path, "label_data_val/"))

    list_tifs = np.sort(os.listdir(tif_path))
    num = 0
    for i in list_tifs:
        if os.path.split(i)[1].rfind("tt") != -1:
            tif = gdal.Open(os.path.join(tif_path, i))
            im_tt = tif.GetRasterBand(1).ReadAsArray()
    min_ = []
    max_ = []
    for j in range(ker, im_tt.shape[0] - ker, step):
        sys.stdout.write("\r{} / {}".format(j, im_tt.shape[0] - ker))
        sys.stdout.flush()
        for i in range(ker, im_tt.shape[1] - ker, step):
            im_tt_part = im_tt[j - ker : j + ker, i - ker : i + ker]
            if np.sum(im_tt_part == 1) < 100:
                continue
            num = num + 1
            im_number = 0
            if np.random.uniform(0, 1) < 0.1:
                save_this_tif_path = "label_data_val/"
            else:
                save_this_tif_path = "label_data_tr/"

            for l in list_tifs:
                if (
                    os.path.split(l)[1].rfind("tt") == -1
                    and os.path.split(l)[1].rfind(".tif") != -1
                    and os.path.split(l)[1].rfind("unet") == -1
                ):
                    im_tr = gdal.Open(os.path.join(tif_path, l))
                    im_tr_full = []
                    form = "GTiff"
                    driver = gdal.GetDriverByName(form)
                    raster_srs = osr.SpatialReference()
                    output = driver.Create(
                        os.path.join(
                            tif_path,
                            "train_data/" + str(num) + "_" + str(im_number) + ".tif",
                        ),
                        2 * ker,
                        2 * ker,
                        im_tr.RasterCount,
                        gdal.GDT_Int16,
                    )
                    raster_srs.ImportFromWkt(tif.GetProjectionRef())
                    output.SetProjection(raster_srs.ExportToWkt())
                    output.SetGeoTransform(tif.GetGeoTransform())
                    for b in range(im_tr.RasterCount):
                        # im_tr_full.append(im_tr.GetRasterBand(b+1).ReadAsArray(i-ker,j-ker,2*ker,2*ker))
                        output.GetRasterBand(b + 1).WriteArray(
                            im_tr.GetRasterBand(b + 1).ReadAsArray(
                                i - ker, j - ker, 2 * ker, 2 * ker
                            )
                        )
                        output.FlushCache()
                    output = None
                    # scipy.misc.imsave(os.path.join(tif_path,'train_data/'+str(num)+'_' + str(im_number) +'.tif'), np.array(im_tr_full).transpose(1,2,0))
                    im_number += 1
            # scipy.misc.imsave(os.path.join(tif_path,'label_data_tr/'+str(num)+'_mask.tif'), im_tt_part)
            form = "GTiff"
            driver = gdal.GetDriverByName(form)
            raster_srs = osr.SpatialReference()
            output = driver.Create(
                os.path.join(tif_path, save_this_tif_path + str(num) + "_mask.tif"),
                2 * ker,
                2 * ker,
                1,
                gdal.GDT_Byte,
            )
            raster_srs.ImportFromWkt(tif.GetProjectionRef())
            output.SetProjection(raster_srs.ExportToWkt())
            output.SetGeoTransform(tif.GetGeoTransform())
            output.GetRasterBand(1).WriteArray(im_tt_part)
            output.FlushCache()
            output = None

    print("\nReading has been Finished!")
    print(round(time.time() - time0))


def map_creating(load_my_model, map_path, tif_path, ker, label_encoder):
    time0 = time.time()
    if load_my_model != "":
        # model = load_model(load_my_model, custom_objects={'masked_categorical_crossentropy': masked_categorical_crossentropy, 'masked_accuracy':masked_accuracy})
        model = load_model(
            load_my_model,
            custom_objects={
                "binary_crossentropy_plus_jaccard_loss": sm.losses.bce_jaccard_loss,
                "jaccard_loss": jaccard_loss,
                "masked_accuracy": masked_accuracy,
            },
            compile=False,
        )

    list_tifs = np.sort(os.listdir(map_path))
    for i in list_tifs:
        if os.path.split(i)[1].rfind("tt") != -1:
            path_for_saving = os.path.join(tif_path, i)
            tif = gdal.Open(os.path.join(tif_path, i))
            im_tt = tif.GetRasterBand(1).ReadAsArray()

    im_res = np.zeros_like(im_tt)
    im_tt = np.pad(im_tt, pad_width=int(ker / 2), mode="constant", constant_values=0)
    xsize = tif.RasterXSize
    ysize = tif.RasterYSize
    result_path = path_for_saving.replace("tt.tif", "unet_map.tif")
    form = "GTiff"
    driver = gdal.GetDriverByName(form)
    raster_srs = osr.SpatialReference()
    output = driver.Create(os.path.join(result_path), xsize, ysize, 1, gdal.GDT_Byte)
    raster_srs.ImportFromWkt(tif.GetProjectionRef())
    output.SetProjection(raster_srs.ExportToWkt())
    output.SetGeoTransform(tif.GetGeoTransform())

    for j in range(ker, im_tt.shape[0] - 2 * ker, ker):
        j = min(j, im_tt.shape[0] - ker)
        sys.stdout.write("\r{} / {}".format(j, im_tt.shape[0] - ker))
        sys.stdout.flush()
        for i in range(ker, im_tt.shape[1] - 2 * ker, ker):
            i = min(i, im_tt.shape[1] - ker)
            im_number = 0
            im_tr_full = []
            for l in list_tifs:
                if (
                    os.path.split(l)[1].rfind("tt") == -1
                    and os.path.split(l)[1].rfind(".tif") != -1
                    and os.path.split(l)[1].rfind("unet") == -1
                ):
                    im_tr = gdal.Open(os.path.join(tif_path, l))
                    for b in range(im_tr.RasterCount):
                        im_tr_full.append(
                            im_tr.GetRasterBand(b + 1).ReadAsArray(
                                i - ker, j - ker, 2 * ker, 2 * ker
                            )
                        )
            part = np.array(im_tr_full).transpose(1, 2, 0)
            #                part = (part-[ 786.,  614.,  427.,  337.]*11)/([ 4405.,  4500.,  4772.,  5488.]*11)
            reconstructed = model.predict(part.reshape((1, 2 * ker, 2 * ker, -1)))
            temp = label_encoder.inverse_transform(
                np.argmax(reconstructed[0], axis=2).reshape(
                    -1,
                )
            ).reshape((2 * ker, 2 * ker))
            im_res[
                j - int(ker / 2) : j + int(ker / 2), i - int(ker / 2) : i + int(ker / 2)
            ] = temp[int(ker / 2) : int(3 * ker / 2), int(ker / 2) : int(3 * ker / 2)]
        output.GetRasterBand(1).WriteArray(im_res)
        output.FlushCache()

    # -------------------------------voting------------------------------------
    #        from scipy import stats
    #
    #        for i in range(ker, im_tr.shape[0]-ker, ker):
    #            #i = min(i, im_tr.shape[0]-ker)
    #            sys.stdout.write('\r{} / {}'.format(i, im_tr.shape[0]-ker))
    #            sys.stdout.flush()
    #            for j in range(ker, im_tr.shape[1]-ker, ker):
    #                #j = min(j, im_tr.shape[1]-ker)
    #                part = im_tr[i-ker:i+ker,j-ker:j+ker]
    #                part[part>16]=9
    #                part = onehot_encoder.transform(label_encoder.transform(part.reshape(-1,)).reshape(-1,1)).reshape((1, 2*ker, 2*ker, -1))
    #                reconstructed = model.predict(part)
    #                reconstructed = label_encoder.inverse_transform(np.argmax(reconstructed[0],axis=2))
    #                print(i,j)
    #                print(reconstructed.shape)
    #                im_res[i-ker:i,j-ker:j,0] = reconstructed[:ker,:ker]
    #                im_res[i:i+ker,j-ker:j,1] = reconstructed[ker:2*ker,:ker]
    #                im_res[i-ker:i,j:j+ker,2] = reconstructed[:ker,ker:2*ker]
    #                im_res[i:i+ker,j:j+ker,3] = reconstructed[ker:2*ker,ker:2*ker]
    #
    #            output.GetRasterBand(1).WriteArray(stats.mode(im_res, axis=2)[0][:,:,0])
    #            output.FlushCache()
    # ---------------------------------------------------------------------------

    output = None
    print("\nCreating Map has been Finished!")
    print(round(time.time() - time0))
