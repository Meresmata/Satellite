import argparse
import os
import typing as tp
import warnings
from functools import partial
from zipfile import ZipFile

import geopandas
import geopy.point as p
import numpy as np
import pandas as pd
import pyproj
from PIL import Image
from geopy.distance import geodesic
from satpy import find_files_and_readers
from satpy.scene import Scene
from sentinelsat import SentinelAPI
from shapely.geometry import box, shape
from shapely.ops import transform
from shapely.wkt import loads

from h3_resolution import h3_res


def download_best(_box: box, download_path: str, user: str, pw: str) -> tp.List[str]:
    _api = SentinelAPI(user, pw, 'https://scihub.copernicus.eu/dhus')

    file_path = os.path.join(download_path, "save.csv")

    if not os.path.exists(file_path):

        products = _api.query(_box,
                              date=('NOW-1MONTH', 'NOW'),
                              platformname='Sentinel-2',
                              processinglevel='Level-1C',
                              cloudcoverpercentage=(0, 10),
                              )

        products_df = _api.to_dataframe(products)

        tile_ids = []

        def _unknown_tile_id(x: str, t_ids: tp.List) -> bool:
            ret_val = x in t_ids
            if not ret_val:
                t_ids.append(x)

            return not ret_val

        # sort products
        products_df_sorted = products_df.sort_values(["cloudcoverpercentage"], ascending=[True])

        # sort out tiles double tiles with higher cloud coverage
        first_tiles = [_unknown_tile_id(x, tile_ids) for x in list(products_df_sorted['tileid'].array)]
        #  first_titles = np.vectorize(_unknown_tile_id(lambda x:x, tile_ids))(products_df_sorted['tileid'].array)
        products_df_sorted_unique = products_df_sorted[first_tiles]

        if not os.path.exists(download_path):
            os.makedirs(download_path)
        products_df_sorted_unique.to_csv(file_path)
    else:
        products_df_sorted_unique = pd.read_pickle(file_path)

    products_df_sorted_unique['area'] = [__estimate_area(loads(e)) for e in
                                         list(products_df_sorted_unique['footprint'].array)]

    #  sort out areas smaller than three quarter of the full size of 100 km * 100 km
    products_df_sorted_unique_larger = products_df_sorted_unique[
        products_df_sorted_unique['area'] > 100000 * 100000 / 4 * 3]

    _api.download_all(products_df_sorted_unique_larger.uuid, download_path)

    # estimate area from footprint

    return [os.path.join(download_path, x) for x in products_df_sorted_unique.title]


def __estimate_area(s: shape) -> float:
    #  TODO testing
    proj = partial(pyproj.transform, pyproj.Proj(init="epsg:4326"), pyproj.Proj(init="epsg:3857"))

    return transform(proj, s).area


def unzip_maps(folder_names: tp.List[str]) -> tp.List[str]:
    # unzip the folders
    zip_files = ["{}.zip".format(name) for name in folder_names if not os.path.exists(name)]
    for file in zip_files:
        with ZipFile("{}.zip".format(file), 'r') as zipObj:
            zipObj.extractall(file)

    return folder_names


def unzip_all_maps(download_dir: str) -> tp.List[str]:
    folder_names = [file.rsplit(".", 1)[0] for file in os.listdir(download_dir) if file.endswith(".zip")]
    # unzip the folders
    full_names = [os.path.join(download_dir, file) for file in folder_names]
    sub_paths = [file for file in full_names if not os.path.exists(file)]
    for file in sub_paths:
        with ZipFile("{}.zip".format(file), 'r') as zipObj:
            zipObj.extractall(file)

    return full_names


def create_coordinate(start_coord: tp.Tuple[float, float], x_offset: float, y_offset: float) -> tp.Tuple[float, float]:
    """
    create a coordinate from a given starting point by shifting it a given x_offset in km and y_offset in km
    :param start_coord: tp.Tuple[float, float]
    :param x_offset: float, length in km
    :param y_offset: float, length in km
    :return: tp.Tuple[float, float]
    """
    start = p.Point(*start_coord)

    dy = geodesic(kilometers=y_offset)
    dx = geodesic(kilometers=x_offset)

    final = dy.destination(dx.destination(start, bearing=90), bearing=0)  # 90 = East, 0 = North...
    return final.longitude, final.latitude


def crop_image_by_coords(_scn: tp.Any, _box: tp.Tuple[float, float, float, float], save_path: str) -> None:
    """
    Crop the satellite Scene to a given Tuple and save that as an image
    :param _scn: Scene
    :param _box: polygon
    :param save_path: str
    :return: None
    """
    scene_llbox = _scn.crop(xy_bbox=_box)

    filename = os.path.join(save_path, 'RGB_{}.tif'.format(str([int(c) for c in _box])))

    scene_llbox.save_dataset('true_color', filename, writer='simple_image', fill_value=0)


def create_image(path: tp.Optional[str] = None) -> (Scene, tp.Optional[str]):
    """
    Create image of the given satellite data of a SentinelSat-2 satellite
    :param path: Path to the raw satellite data
    :return: (Scene, str), Scene of the satellite image, full path to the .tif create
    """
    files = find_files_and_readers(base_dir=path, reader='msi_safe')

    _scn = Scene(filenames=files)
    _scn.load(['true_color'])

    filename = None
    if path is not None:
        filename = os.path.join(path, 'RGB.tif')
        if not os.path.exists(filename):
            _scn.save_dataset('true_color', filename, writer='simple_image', fill_value=0)
    return _scn, filename


def create_xy_bbox(_scn: Scene, xy_dist: float) -> tp.Tuple[tp.List, tp.List]:
    """
    Prepare the boxes for to crop the satellite images.
    :param _scn: Scene
    :param xy_dist: float, distance in meter of min x,y length of box
    :return: tp.Tuple[tp.List, tp.List] list of tuples of 1. pixels (left, upper, right, lower)
    and 2. coordinates (minx, miny, maxx, maxy)
    """
    xs = _scn['true_color'].attrs['area'].projection_x_coords
    ys = _scn['true_color'].attrs['area'].projection_y_coords
    step = int(np.where(xs > xs[0] + xy_dist)[0][0]) + 1

    pixel_boxes = []
    coord_boxes = []
    for x in range(0, len(xs) - step, step):
        for y in range(0, len(ys) - step, step):
            # minx, miny, maxx, maxy
            coord_boxes.append((int(xs[x]), int(ys[y + step - 1]), int(xs[x + step - 1]), int(ys[y])))
            # left, upper, right, lower
            # images have a slight off-set of to the upper code,
            # but else wise the pixel size would be smaller
            pixel_boxes.append((x, y, x + step, y + step))

    return pixel_boxes, coord_boxes


def crop_image_by_points(im: Image,
                         _pixels: tp.Tuple[int, int, int, int],
                         _coords: tp.Tuple[int, int, int, int] = None,
                         save_path: tp.Optional[str] = None) -> tp.Tuple[Image.Image, str]:
    """
    Crop and save the image using the points, naming scheme ouf output file similar to crop_image_by_box
    :param im:Image, pil Image object
    :param save_path:str, full path to the image
    :param _coords: tp.Tuple[int, int, int, int] [minx, miny, maxx, maxy]
    :param _pixels:tp.Tuple[int, int, int, int], (left, upper, right, lower) of pixels
    :return:(Image.Image, str), cropped image and optional name of the cropped image, as long save_path is specified
    """
    crop_im = im.crop(_pixels)
    if save_path:
        filename = os.path.join(save_path, 'RGB_{}.tif'.format(str(_coords)))
        crop_im.save(filename, "PNG")
    else:
        filename = None

    return crop_im, filename


def box_of_nation(nation: str) -> str:
    """
    Create a shapely box for a given Country
    :param nation: str
    :return: Polygon
    """
    _world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    test_str = 'name=="{}"'.format(nation)
    nation = _world.query(test_str)

    return box(*nation.geometry.total_bounds)


def shape_of_nation(nation: str) -> str:
    """
    Create a shapely shape for a given Country
    :param nation: str
    :return: Polygon, ?
    """
    _world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    test_str = 'name=="{}"'.format(nation)
    nation = _world.query(test_str)

    return nation.geometry


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='Path to the main folder of files.')
    parser.add_argument('user', type=str)
    parser.add_argument('password', type=str)
    # for a he resolution of about 7 https://uber.github.io/h3/#/documentation/core-library/resolution-table
    # edge length
    # a * 0.866 is circa the radius
    parser.add_argument('-r', '--resolution', type=int, default=7)
    parser.add_argument('-p', '--percentage', type=int, default=100)

    # create training set
    # Taichung, Regensburg, Shenzhen, Ürümqi, Berchtesgaden, Shanghai (sea, mountains, desert, city)
    coords = [(24.1, 120.7), (49.0, 12.0), (22.5, 114.1), (43.8, 87.6), (47.6, 13.0), (31.22, 121.46)]

    args = parser.parse_args()

    p_dir = args.path
    user_name = args.user
    password = args.password
    r = float("{:04.2f}".format(h3_res[args.resolution] * 0.866))

    for coord in coords:
        box_pos = box(*create_coordinate(coord, -r, -r), *create_coordinate(coord, r, r))

        download_best(box_pos, p_dir, user_name, password)

        # for nation box  build up raster, then filter out elements not in shape of the nation
        # for raster in image do classification

    for raw_map in unzip_maps(p_dir):
        scn, o_path = create_image(raw_map)

        image = Image.open(o_path)

        pixels_list, coords_list = create_xy_bbox(scn, 2 * r * 1000)
        for pxls, crds in zip(pixels_list, coords_list):
            crop_image_by_points(image, pxls, crds, raw_map)
