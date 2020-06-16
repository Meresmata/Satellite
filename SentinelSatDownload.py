import argparse
import os
import typing as tp
import warnings
from functools import partial
from zipfile import ZipFile

import geopandas
import pandas as pd
import geopy.point as p
import numpy as np
import pyproj
from PIL import Image
from geopy.distance import geodesic
from satpy import find_files_and_readers
from satpy.scene import Scene
from sentinelsat import SentinelAPI
from shapely.geometry import box, Polygon, LineString
from shapely.ops import transform, split
from shapely.wkt import loads

from h3_resolution import h3_radius


def split_polygon(polygon: Polygon) -> tp.List[Polygon]:
    """
    half the polygons so long until they are smaller than the size of two sentinel2 sat tiles
    :param polygon: Polygon to be split
    :return: list of splitted polygons
    """
    polys = []

    if __estimate_area(polygon) > 100000 * 100000 // 2:
        # split horizontal or vertical
        split_vertical = polygon.bounds[3] - polygon.bounds[1] > polygon.bounds[2] - polygon.bounds[0]
        line: LineString
        if split_vertical:
            line = LineString([(polygon.bounds[0], polygon.centroid.bounds[1]),
                               (polygon.bounds[2], polygon.centroid.bounds[1])])
        else:
            line = LineString([(polygon.centroid.bounds[0], polygon.bounds[1]),
                               (polygon.centroid.bounds[0], polygon.bounds[3])])

        new = split(polygon, line)
        new = [split_polygon(e) for e in new]
        [polys.append(e) for li in new for e in li]
    else:
        polys.append(polygon)

    return polys


def download_best(boxes: tp.List[Polygon], download_path: str, user: str, pw: str) -> tp.List[str]:
    if boxes is None:
        raise AttributeError()

    _api = SentinelAPI(user, pw, 'https://scihub.copernicus.eu/dhus')
    products_dataframes = []
    for _box in boxes:

        products = _api.query(_box,
                              date=('NOW-1YEAR', 'NOW'),
                              platformname='Sentinel-2',
                              processinglevel='Level-1C',
                              cloudcoverpercentage=(0, 10),
                              )
        products_df = _api.to_dataframe(products)

        # sort products
        products_df_sorted = products_df.sort_values(["cloudcoverpercentage"], ascending=[True])

        footprint = [loads(s) for s in products_df_sorted['footprint']]
        products_df_sorted['area'] = list(map(__estimate_area, footprint))

        #  sort out areas smaller than three quarter of the full size of 100 km * 100 km
        min_area = 100000 * 100000 // 20 * 17  # 85 %
        products_df_sorted_larger = products_df_sorted[products_df_sorted['area'] > min_area]
        products_df_sorted_larger = products_df_sorted_larger.groupby('tileid').head(1)
        products_dataframes.append(products_df_sorted_larger.head(3))

    products_df = pd.concat(products_dataframes)
    products_df = products_df.groupby(['uuid']).head(1)

    _api.download_all(products_df.uuid, download_path)

    # estimate area from footprint
    return [os.path.join(download_path, x) for x in products_df.title]


def __estimate_area(s: Polygon) -> float:
    proj = partial(pyproj.transform, pyproj.Proj(init="epsg:4326"), pyproj.Proj(init="epsg:3395"))
    area = transform(proj, s).area
    return area


def unzip_maps(folder_names: tp.List[str]) -> tp.List[str]:
    # unzip the folders
    # create short names/tile names - short names need for max. path length restrictions, while unzipping
    # list of old_name and short_name paths
    folder_names = [[name, os.path.join(os.path.dirname(name), os.path.basename(name).split("_")[5])] for name in
                    folder_names if not os.path.exists(name)]
    zip_files = [name for name in folder_names if not os.path.exists(name[1])]

    for file in zip_files:
        with ZipFile("{}.zip".format(file[0]), 'r') as zipObj:
            zipObj.extractall(file[1])

    return [name[1] for name in folder_names]


def unzip_all_maps(download_dir: str) -> tp.List[str]:
    folder_names = [os.path.join(download_dir, file.rsplit(".", 1)[0]) for file in os.listdir(download_dir) if
                    file.endswith(".zip")]
    # unzip the folders
    return unzip_maps(folder_names)


def create_coordinate(start_coord: tp.Tuple[float, float], x_offset: float, y_offset: float) -> tp.Tuple[float, float]:
    """
    create a coordinate from a given starting point by shifting it a given x_offset in km and y_offset in km
    :param start_coord: tp.Tuple[float, float]
    :param x_offset: float, length in km
    :param y_offset: float, length in km
    :return: tp.Tuple[float, float]
    """
    start = p.Point(*start_coord)

    dy = geodesic(meters=y_offset)
    dx = geodesic(meters=x_offset)

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


def create_image(path: str) -> (Scene, tp.Optional[str]):
    """
    Create image of the given satellite data of a SentinelSat-2 satellite
    :param path: Path to the raw satellite data
    :return: (Scene, str), Scene of the satellite image, full path to the .tif create
    """
    files = find_files_and_readers(base_dir=path, reader='msi_safe')

    _scn = Scene(filenames=files)
    _scn.load(['true_color'])

    filename = os.path.join(path, 'RGB.tif')
    if not os.path.exists(filename):
        _scn.save_dataset('true_color', filename, writer='simple_image', fill_value=0)
    return _scn, filename


def create_xy_bbox(_scn: Scene, xy_dist: float) -> tp.Tuple[tp.List[tp.Tuple[int, int, int, int]],
                                                            tp.List[tp.Tuple[int, int, int, int]]]:
    """
    Prepare the boxes for to crop the satellite images.
    :param _scn: Scene
    :param xy_dist: float, distance in meter of min x,y length of box
    :return: tp.Tuple[tp.List, tp.List] list of tuples of 1. pixels (left, upper, right, lower)
    and 2. coordinates (minx, miny, maxx, maxy)
    """
    xs = _scn['true_color'].attrs['area'].projection_x_coords.astype(int)
    ys = _scn['true_color'].attrs['area'].projection_y_coords.astype(int)
    step = int(np.where(xs > xs[0] + xy_dist)[0][0]) + 2

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


def box_of_nation(nation: str) -> Polygon:
    """
    Create a shapely box for a given Country
    :param nation: str
    :return: Polygon
    """
    _world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    test_str = 'name=="{}"'.format(nation)
    nation = _world.query(test_str)

    return box(*nation.geometry.total_bounds)


def shape_of_nation(nation: str) -> Polygon:
    """
    Create a shapely shape for a given Country
    :param nation: str
    :return: Polygon, ?
    """
    _world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    test_str = 'name=="{}"'.format(nation)
    nation = _world.query(test_str)

    return nation.geometry.unary_union


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
    r = float("{:06.0f}".format(h3_radius(args.resolution)))

    for coord in coords:
        box_pos = box(*create_coordinate(coord, -r, -r), *create_coordinate(coord, r, r))

        download_best(box_pos, p_dir, user_name, password)

        # for nation box  build up raster, then filter out elements not in shape of the nation
        # for raster in image do classification

    for raw_map in unzip_maps(p_dir):
        scn, o_path = create_image(raw_map)

        image = Image.open(o_path)

        pixels_list, coords_list = create_xy_bbox(scn, 2 * r)
        for pxls, crds in zip(pixels_list, coords_list):
            crop_image_by_points(image, pxls, crds, raw_map)
