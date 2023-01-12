import argparse
import os

import fiona
import gdal
import numpy as np
import rasterio
import rasterio.mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("im_path", type=str, help="image path")
    parser.add_argument("shp_path", type=str, help="shp path")

    opt = parser.parse_args()

    im_path = opt.im_path
    shp_path = opt.shp_path
    res_path = os.path.join(os.path.split(im_path)[0], "tt.tif")
    g = gdal.Open(im_path)

    b = g.GetRasterBand(1).ReadAsArray().astype(np.byte)
    drv = gdal.GetDriverByName("GTiff")
    dst_ds = drv.Create(
        res_path,
        g.RasterXSize,
        g.RasterYSize,
        1,
        gdal.GDT_Byte,
        options=["COMPRESS=LZW"],
    )
    dst_ds.SetGeoTransform(g.GetGeoTransform())
    dst_ds.SetProjection(g.GetProjection())

    with fiona.open(shp_path, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]

    with rasterio.open(im_path) as src:
        out_image, _, _ = rasterio.mask.raster_geometry_mask(src, shapes, crop=False)

    print(out_image.shape)
    dst_ds.GetRasterBand(1).WriteArray(out_image.astype(np.byte) + 1)
    dst_ds = None
