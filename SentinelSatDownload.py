import argparse
import os
import typing as tp
import warnings
from zipfile import ZipFile

import geopandas
import geopy.point as p
import numpy as np
from PIL import Image
from geopy.distance import geodesic
from pyresample.geometry import AreaDefinition
from satpy import find_files_and_readers
from satpy.scene import Scene
from sentinelsat import SentinelAPI
from shapely.geometry import box
from tqdm import tqdm

from h3_resolution import h3_res


def download_best(_box: box, download_path: str, user: str, pw: str):
    _api = SentinelAPI(user, pw, 'https://scihub.copernicus.eu/dhus')

    products = _api.query(_box,
                          date=('20200101', 'NOW'),
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

    first_tiles = [_unknown_tile_id(x, tile_ids) for x in list(products_df_sorted['tileid'].array)]
    products_df_sorted_unique = products_df_sorted[first_tiles]

    _api.download_all(products_df_sorted_unique.head(1).uuid, download_path)


def unzip_maps(download_dir: str) -> tp.List[str]:
    folder_names = [file.rsplit(".", 1)[0] for file in os.listdir(download_dir) if file.endswith(".zip")]
    # unzip the folders
    sub_paths = [os.path.join(download_dir, file) for file in folder_names]
    for file in sub_paths:
        with ZipFile("{}.zip".format(file), 'r') as zipObj:
            zipObj.extractall(file)

    return sub_paths


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

    filename = os.path.join(save_path, 'RGB_{}.png'.format(str([int(c) for c in _box])))

    scene_llbox.save_dataset('true_color', filename, writer='simple_image', fill_value=0)


def create_image(path: str) -> (Scene, str):
    """
    Create image of the given satellite data of a SentinelSat-2 satellite
    :param path: Path to the raw satellite data
    :return: (Scene, str), Scene of the satellite image, full path to the .png create
    """
    files = find_files_and_readers(base_dir=path, reader='msi_safe')

    _scn = Scene(filenames=files)
    _scn.load(['true_color'])

    filename = os.path.join(path, 'RGB.png')
    _scn.save_dataset('true_color', filename, writer='simple_image', fill_value=0)
    return _scn, filename


def create_xy_bbox(_scn: Scene, xy_dist: float, is_slicing_coordinates: bool = True) -> tp.List:
    """
    Prepare the boxes for to crop the satellite images.
    :param _scn: Scene
    :param xy_dist: float, distance in meter of min x,y length of box
    :param is_slicing_coordinates: bool, decide return the pixels, or coordinates
    :return: tp:list list of Shapely boxes
    """
    xs = _scn['true_color'].attrs['area'].projection_x_coords
    ys = _scn['true_color'].attrs['area'].projection_y_coords
    step = int(np.where(xs > xs[0] + xy_dist)[0][0]) + 1

    ret_boxes = []
    for x in range(0, len(xs) - step, step):
        for y in range(0, len(ys) - step, step):
            try:
                if is_slicing_coordinates:
                    # minx, miny, maxx, maxy
                    ret_boxes.append((xs[x], ys[y + step - 1], xs[x + step - 1], ys[y]))
                else:
                    # left, upper, right, lower
                    # images have a slight off-set of to the upper code,
                    # but elsewhise the pixel size would be smaller
                    ret_boxes.append((x, y, x + step, y + step))
            except TypeError:
                print("TypeError")

    return ret_boxes


def crop_image_by_points(im: Image, area: AreaDefinition,
                         xy_points: tp.Tuple[int, int, int, int],
                         save_path: tp.Optional[str] = None) -> tp.Tuple[Image.Image, str]:
    """
    Crop and save the image using the points, naming scheme ouf output file similar to crop_image_by_box
    :param im:Image, pil Image object
    :param save_path:str, full path to the image
    :param area: AreaDefinition, to calculate the  (minx, miny, maxx, maxy) equals (left, lower, right, upper) pf coordinates
    :param xy_points:tp.Tuple[int, int, int, int], (left, upper, right, lower) of pixels
    :return:(Image.Image, str), cropped image and optional name of the cropped image, as long save_path is specified
    """
    crop_im = im.crop(xy_points)
    if save_path:
        x_s = area.projection_x_coords
        y_s = area.projection_y_coords
        xy_coords = (x_s[xy_points[0]], y_s[xy_points[3]], x_s[xy_points[2]], y_s[xy_points[1]])
        filename = os.path.join(save_path, 'RGB_{}.png'.format(str([int(c) for c in xy_coords])))
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
    # Taichung, Regensburg, Xiamen, Shenzhen, Ürümqi, Berchtesgaden, Beijing, Shanghai (sea, mountains, desert, city)
    coords = [(24.1, 120.7), (49.0, 12.0), (24.5, 118.1), (22.5, 114.1), (43.8, 87.6), (47.6, 13.0), (39.91, 116.40),
              (31.22, 121.46)]

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
        from_coord = False
        boxes = create_xy_bbox(scn, 2 * r * 1000, from_coord)
        b_bar = tqdm(boxes)
        for xy_box in b_bar:
            if from_coord:
                crop_image_by_coords(scn, xy_box, raw_map)
            else:
                crop_image_by_points(image, scn['true_color'].attrs['area'], xy_box, raw_map)
