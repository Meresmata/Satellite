import argparse
import typing as tp
from shapely.geometry import Polygon, box
import osm
import os
import requests
import zipfile
import io
import rasterio as rio


def download_tiles(bbox: Polygon, path: tp.Optional[str] = None):
    northing_min, easting_min, northing_max, easting_max = [(int(x)//20) * 20 for x in bbox.bounds]

    for y in range(northing_max + 20, northing_min,  -20):
        for x in range(easting_max + 20, easting_min, -20):
            northing_letter = "N" if northing_min >= 0 else "S"
            easting_letter = "E" if easting_min >= 0 else "W"

            zip_file_url = r"https://s3-eu-west-1.amazonaws.com/vito.landcover.global/2015/{}{}{}{}_ProbaV_LC100_epoch2015_global_v2.0.2_products_EPSG-4326.zip".format(easting_letter, x, northing_letter, y)

            r = requests.get(zip_file_url, stream=True)
            z = zipfile.ZipFile(io.BytesIO(r.content))

            full_path = os.path.join(path, zip_file_url.split(r"/")[-1].rsplit(".", 1)[0])
            z.extractall(full_path)

            yield full_path


def load_urban_raster(path: str, tile_box: tp.Tuple[float, float]):
    easting_min, northing_min = [(int(x) // 20) * 20 for x in tile_box]
    northing_letter = "N" if northing_min > 0 else "S"
    easting_letter = "E" if easting_min > 0 else "W"

    mask_name = "{}{}{}{}_ProbaV_LC100_epoch2015_global_v2.0.2_urban-coverfraction-layer_EPSG-4326.tif".format(easting_letter, easting_min, northing_letter, northing_min)

    full_path = os.path.join(path, mask_name)
    return rio.open(full_path, driver='GTiff')


def mask_contains(r, p: tp.Any):
    if type(p) is tuple:
        idx = r.index(*p)
        value = int(r.read(1)[idx])
        return value == 100

    elif type(p) is list:
        return [v == 255 for v in r.sample(p)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('nation', type=str, help='Path to the train main folder of files.')
    parser.add_argument('-p', '--path', type=str, default=None)

    args = parser.parse_args()
    p_dir = args.path
    nation = args.nation

    s = osm.get_osm_national_boundary(nation)
    tile = next(download_tiles(box(*s.bounds), p_dir))

    # tile = "/media/muemmel/6532-6362/E120N40_ProbaV_LC100_epoch2015_global_v2.0.2_products_EPSG-4326"

    raster = load_urban_raster(tile, (120, 40))
    is_urban = mask_contains(raster, (121.5, 25.1))
