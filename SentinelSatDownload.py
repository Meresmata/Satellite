import argparse
import os
import typing as tp
import warnings
from random import randint
from zipfile import ZipFile

import geopandas
import geopy.point as p
from geopy.distance import geodesic
from satpy import find_files_and_readers
from satpy.scene import Scene
from sentinelsat import SentinelAPI
from shapely.geometry import polygon, box
from tqdm import tqdm

from h3_resolution import h3_res


def create_coordinate(start_coord: tp.Tuple[float, float], x_offset: float, y_offset: float) -> tp.Tuple[float, float]:
    start = p.Point(*start_coord)

    dy = geodesic(kilometers=y_offset)
    dx = geodesic(kilometers=x_offset)

    final = dy.destination(dx.destination(start, bearing=90), bearing=0)  # 90 = East, 0 = North...
    return final.longitude, final.latitude


def crop_image_by_box(scn: Scene, _box: polygon, out_path: str) -> None:
    scene_llbox = scn.crop(xy_bbox=_box.bounds)

    filename = os.path.join(out_path, 'RGB_{}.png'.format(str([int(c) for c in _box.bounds])))

    scene_llbox.save_dataset('true_color', filename)


def create_image(path: str) -> Scene:
    files = find_files_and_readers(base_dir=path, reader='msi_safe')

    scn = Scene(filenames=files)
    scn.load(['true_color'])

    filename = os.path.join(path, 'RGB.png')
    scn.save_dataset('true_color', filename)
    return scn


def create_xy_bbox(scn: Scene, xy_dist: float) -> tp.List:
    xs = sorted(scn['true_color'].attrs['area'].projection_x_coords)
    ys = sorted(scn['true_color'].attrs['area'].projection_y_coords)

    xs_slices = [xs[0]]
    ys_slices = [ys[0]]

    x = xs[0]
    y = ys[0]

    def _slicing(c, l_a, l_b):
        maxi = l_a[-1]
        while c < maxi:
            c = c + xy_dist
            for c_inner in l_a:
                if c_inner > c:
                    c = c_inner
                    l_b.append(c)
                    break

    _slicing(x, xs, xs_slices)
    _slicing(y, ys, ys_slices)
    boxes = []
    for x in range(len(xs_slices) - 1):
        for y in range(len(ys_slices) - 1):
            try:
                boxes.append(box(xs_slices[x], ys_slices[y], xs_slices[x + 1], ys_slices[y + 1]))
            except TypeError:
                print("TypeError")

    boxes = [b for b in boxes if b.within(box(*scn['true_color'].attrs['area'].area_extent))]

    return boxes


def box_of_nation(nation: str) -> str:
    """
   create a shapely box for a given Country
    :param nation: str
    :return: str
    """
    _world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    test_str = 'name=="{}"'.format(nation)
    nation = _world.query(test_str)

    return box(*nation.geometry.total_bounds)


def shape_of_nation(nation: str) -> str:
    """
    create a shapely shape for a given Country
    :param nation: str
    :return: str
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
    # Taichung, Regensburg, Xiamen, Shenzhen, Ürümqi, Berchtesgaden (sea, mountains, desert)
    coords = [(24.1, 120.7), (49.0, 12.0), (24.5, 118.1), (22.5, 114.1), (43.8, 87.6), (47.6, 13.0)]

    args = parser.parse_args()

    p_dir = args.path
    user = args.user
    password = args.password
    r = float("{:04.2f}".format(h3_res[args.resolution] * 0.866))
    percentage = args.percentage

    api = SentinelAPI(user, password, 'https://scihub.copernicus.eu/dhus')

    for coord in coords:
        box_pos = box(*create_coordinate(coord, -r, -r), *create_coordinate(coord, r, r))

        products = api.query(box_pos,
                             date=('20200101', 'NOW'),
                             platformname='Sentinel-2',
                             processinglevel='Level-1C',
                             cloudcoverpercentage=(0, 10),
                             )

        products_df = api.to_dataframe(products)

        tile_ids = []


        def _unknown_tile_id(x: str, t_ids: tp.List) -> bool:

            ret_val = x in t_ids
            if not ret_val:
                t_ids.append(x)

            return not ret_val


        # sort products
        products_df_sorted = products_df.sort_values(["cloudcoverpercentage"], ascending=[True])
        # products_df_sorted.to_csv(products_df_sorted.to_csv("F:data.csv"), index=False, sep=";")

        first_tiles = [_unknown_tile_id(x, tile_ids) for x in list(products_df_sorted['tileid'].array)]
        products_df_sorted_unique = products_df_sorted[first_tiles]

        api.download_all(products_df_sorted_unique.head(1).uuid, p_dir)

        # for nation box  build up raster, then filter out elements not in shape of the nation
        # for raster in image do classification

    folder_names = [file.rsplit(".", 1)[0] for file in os.listdir(p_dir) if file.endswith(".zip")]
    # unzip the folders
    sub_paths = [os.path.join(p_dir, file) for file in folder_names]
    for file in sub_paths:
        with ZipFile("{}.zip".format(file), 'r') as zipObj:
            zipObj.extractall(file)

    p_bar = tqdm(sub_paths)
    for folder in p_bar:
        p_bar.set_description("Processing {}".format(folder))
        img = create_image(folder)

        b_bar = tqdm([x for x in create_xy_bbox(img, 2 * r * 1000) if randint(0, 100) < percentage])
        for xy_box in b_bar:
            crop_image_by_box(img, xy_box, folder)
