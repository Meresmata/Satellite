import argparse
import os
import typing as tp
import warnings
import gc
from functools import partial

import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import pyproj.crs
from PIL import Image, ImageDraw
from satpy import scene
from shapely.geometry import box, Polygon
from shapely.ops import transform


import SentinelSatDownload as Dwn
import classificator as clf
from h3_resolution import h3_radius


class Patch:
    """
    represents a patch of a satellite map of a given size, that holds a distinct classifier
    for the raster analysis of the map
    """

    def __init__(self, pixels: tp.Tuple[int, int, int, int], _coords: tp.Tuple[int, int, int, int], map_system: str):
        """
        making the patch somewhat smaller to create a check board looking overlay
        :param pixels:tp.Tuple[int, int, int, int], (left, upper, right, lower) of pixels
        :param _coords: tp.Tuple[int, int, int, int]
        :param map_system: str, epsg, code
        """
        self.p_minx = pixels[0] + 4
        self.p_maxx = pixels[2] - 4
        self.p_miny = pixels[1] + 4
        self.p_maxy = pixels[3] - 4
        self.coords = _coords
        self.center = tuple(box(*_coords).centroid.coords)[0]
        self.system = map_system

        self._classifier: tp.Optional[str] = None
        self.name = str("RGB_{}".format([int(c) for c in _coords]))

    def __str__(self):
        return self.name

    def set_classifier(self, _classifier: str):
        """
        setting the _classifier of the of that element of the raster
        :param _classifier: str classifier from the neural network
        :return: Patch
        """
        # if only one classifier or all are equal

        if self._classifier is None or self._classifier == _classifier:
            self._classifier = _classifier
        else:
            self._classifier = "mixed"

        return self

    def get_classifier(self) -> tp.Optional[str]:
        """
        returns the classifier name
        :return: str, the classifier name
        """
        return self._classifier

    def convert_degree(self, point):
        out_proj = pyproj.Proj(init="epsg:4326")
        in_proj = pyproj.Proj(self.system)
        return pyproj.transform(in_proj, out_proj, *point)


class SatelliteMap:
    """
    A given satellite map of a given size, with the range of of  coordinates in a distinct coordinate system
    """

    def __init__(self, _scn: scene, path: str, shape: Polygon = None):
        """
        :param _scn: scene, of the original satellite map
        :param path: tp.Optional[str], if given activates export of export of raster image while classify, or export
        of classifier of local satellite image while export whole image of nation, exported to the given path
        :param shape: shape in degrees
        """
        area = _scn['true_color'].attrs['area']
        self.__minx = area.area_extent[0]
        self.__maxx = area.area_extent[2]
        self.__miny = area.area_extent[1]
        self.__maxy = area.area_extent[3]
        self.img_path = os.path.join(path, "RGB.tif")
        self.scn = _scn
        self.map_path = path

        self.system: str = area.crs.srs  # epsg-code
        self.shape = box(*area.area_extent) if shape is None else transform_shape_to(shape, self.system).intersection(
            box(*area.area_extent))
        self.__p_maxx = area.height
        self.__p_maxy = area.width
        self.patches: tp.Dict[str, Patch] = {}
        self.name = str("RGB_{}".format([int(c) for c in area.area_extent]))

    def add_patch(self, _patch: Patch):
        """
        :param _patch: Patch itself
        :return:
        """
        self.patches[str(_patch)] = _patch

    def create_raster_image(self, image_path: tp.Optional[str] = None):
        """
        create an image with the classification information of the raster
        the class urban in red and other classes (rural) in green
        order is w, h, c?
        :param image_path: tp.Optional[str], path of the file name to be created
        :return: an image at image_path
        """
        _img = np.ones((self.__p_maxx, self.__p_maxy, 3), dtype="uint8")
        _img = _img * 255
        p_minx, p_miny, p_center_long, p_center_lat, p_class = [], [], [], [], []
        for _patch in self.patches.values():
            _cls = _patch.get_classifier()
            # create image color coding
            if _cls == "urban":
                _img[_patch.p_miny:_patch.p_maxy, _patch.p_minx:_patch.p_maxx] = np.array([255, 0, 0])
            elif _cls == "rural":
                _img[_patch.p_miny:_patch.p_maxy, _patch.p_minx:_patch.p_maxx] = np.array([0, 255, 0])
            elif _cls == "water":
                _img[_patch.p_miny:_patch.p_maxy, _patch.p_minx:_patch.p_maxx] = np.array([0, 125, 125])
            elif _cls == "mixed":
                _img[_patch.p_miny:_patch.p_maxy, _patch.p_minx:_patch.p_maxx] = np.array([255, 215, 0])
            elif _cls == "black":
                _img[_patch.p_miny:_patch.p_maxy, _patch.p_minx:_patch.p_maxx] = np.array([0, 0, 0])
            elif _cls == "cloud":
                _img[_patch.p_miny:_patch.p_maxy, _patch.p_minx:_patch.p_maxx] = np.array([0, 255, 255])

            # create df for training
            p_minx.append(int(_patch.coords[0]))
            p_miny.append(int(_patch.coords[1]))
            p_center_lat.append(_patch.center[0])
            p_center_long.append(_patch.center[1])
            p_class.append(_cls)

        # df = pd.DataFrame(data={"x-Coord": p_minx, "y-Coord": p_miny, "Latitude_m": p_center_lat,
        #                         "Longitude_m": p_center_lat, "Classifier": p_class})
        # df.to_csv(image_path.rsplit(".", 1)[0] + ".csv")

        _img = Image.fromarray(_img)

        # add the coordinates of the patch in the image as text
        d = ImageDraw.Draw(_img)
        for _patch in self.patches.values():
            d.text((_patch.p_minx, _patch.p_miny), str(_patch.coords), fill=(0, 0, 255))

        if image_path:
            _img.save(image_path)
        return _img

    def classify(self, resolution: int, cls_dict: tp.Dict[int, str]):

        _r = float("{:06.0f}".format(h3_radius(resolution)))
        pixel_boxes, coord_boxes = Dwn.create_xy_bbox(self.scn, 2 * _r)
        tmp_patches = []
        tmp_images = []
        img = Image.open(self.img_path)
        for p_box, c_box in zip(pixel_boxes, coord_boxes):
            _img, _ = Dwn.crop_image_by_points(img, p_box, c_box)  # raw_map_path
            patch = Patch(p_box, c_box, self.system)
            tmp_images.append(_img)
            tmp_patches.append(patch)

        if len(tmp_images) > 0:
            gc.collect()
            # classifier = clf.get_multi_classifier(tmp_images, model_ps, cls_dict)
            classifier = clf.get_classifier(tmp_images, model_ps[0], cls_dict)
            # temp_dict = {str(p): p.set_classifier(clf.single_classifier(classifier[i]))
            #              for i, p in enumerate(tmp_patches)}
            temp_dict = {str(p): p.set_classifier(clf.single_classifier(classifier[i])) for i, p in enumerate(tmp_patches)}
            self.patches.update(temp_dict)

            for i, p in enumerate(self.patches.values()):
                if p.get_classifier() == "mixed":
                    tmp_images[i].save(os.path.join(p_dir, "mixed", str(p) + ".png"))

        if not os.path.exists(os.path.join(self.map_path, "test.tif")):
            overlay = self.create_raster_image()  # os.path.join(self.map_path, "control.tif")
            blended = Image.blend(img, overlay, 0.15)
            blended.save(os.path.join(self.map_path, "test.tif"))
            self.export_classification()

    def export_classification(self):
        sys = self.system
        exportable = {}
        for p in self.patches.values():
            if len(exportable) > 0:
                exportable["latitude"].append(p.center[0])
                exportable["longitude"].append(p.center[1])
                exportable["classifier"].append(p.get_classifier())
                exportable["crs"].append(sys)
                exportable["id"].append(str(p))
            else:
                exportable = {"latitude": [p.center[0]], "longitude": [p.center[1]],
                              "classifier": [p.get_classifier()], "crs": [sys], "id": [str(p)]}

        df = pd.DataFrame(data=exportable)
        geometry = geopandas.points_from_xy(df.latitude, df.longitude)
        local_frame = geopandas.GeoDataFrame(df, geometry=geometry, crs=pyproj.crs.CRS(sys))
        local_frame = local_frame.to_crs("4326")

        local_frame.to_file(os.path.join(self.map_path, "classifier.geojson"), driver='GeoJSON')
        return local_frame

    def __str__(self):
        return self.name


class NationalMap:

    def __init__(self, _name: str, user_credentials: tp.Tuple[str, str], resolution: int = 7):
        self.box = Dwn.box_of_nation(_name)
        self.__shape = Dwn.shape_of_nation(_name)
        self.maps: tp.Dict[str, SatelliteMap] = {}
        self.__usr_name = user_credentials[0]
        self.__pw = user_credentials[1]
        self.name = _name
        self.resolution_level = resolution
        self.resolution = int(h3_radius(resolution))  # reduce the resolution the scale of meters

    def __str__(self):
        return self.name

    def add_local_maps(self, folder: str, _box: tp.Optional[tp.Tuple] = None):
        _box = _box if _box is not None else self.box

        if not os.path.exists(folder):
            os.makedirs(folder)

        polygons = Dwn.split_polygon(_box)
        _zip_files = Dwn.download_best(polygons, folder, self.__usr_name, self.__pw)

        for file in Dwn.unzip_maps(_zip_files):
            _scn, path = Dwn.create_image(file)

            sat_map = SatelliteMap(_scn, file, self.__shape)

            self.maps[str(sat_map)] = sat_map

    def classify_all(self, cls_dict: tp.Dict):

        delete_dict_keys = []
        for name, _map in self.maps.items():
            # if whole map outside the nation
            if _map.shape.is_empty:
                # remove map from national map list
                delete_dict_keys.append(name)
            else:
                local_frame = os.path.join(".", "{}.geojson".format(self.name))
                if not os.path.exists(local_frame):
                    _map.classify(self.resolution_level, cls_dict)
                else:
                    gf = geopandas.read_file(local_frame)
                    list(map(lambda x, y: _map.patches[x].set_classifier(y), list(gf.name), list(gf.classifier)))

        # remove all maps from national map list, that are outside the mainland
        [self.maps.pop(k) for k in delete_dict_keys]

    def export_classification(self, path: str) -> geopandas.GeoDataFrame:

        # create a geoframe per coordinate system and convert those to EPSG 4326
        # merge those geoframes
        gdfs = []
        for _map in self.maps.values():
            gdfs.append(_map.export_classification())

        gdf = geopandas.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
        gdf.to_file(path, driver='GeoJSON')
        return gdf


def transform_shape_to(s, system):
    proj = partial(pyproj.transform, pyproj.Proj(init="epsg:4326"), pyproj.Proj(system))

    return transform(proj, s)


def paint_classification(geo: geopandas.GeoDataFrame, name: str, save_path):
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    _nation = world.query('name=="{}"'.format(name))

    # Define base of the plot.
    fig, ax = plt.subplots(figsize=(20, 20), dpi=100)

    # Disable the axes
    # ax.set_axis_off()

    geo = geo[geo["classifier"] in ["water", "urban", "rural"]]

    geo.plot(
        column='classifier',  # Column defining the color
        cmap='jet',  # Colormap
        marker='H',  # marker layout. Here a Hexagon.
        ax=ax,  # Base
        markersize=2
    )
    ax.set_title('{} Classification'.format(name), fontsize=65)

    # Plot the boundary of the countries on top
    _nation.geometry.boundary.plot(color=None, edgecolor='black', ax=ax)

    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument('map_path', type=str, help='Path to the main folder were the satellite image shall be saved.')
    parser.add_argument('user', type=str)
    parser.add_argument('password', type=str)
    parser.add_argument('-m', '--model_path', type=str, help='Path to the trained model', action="append")
    parser.add_argument('-r', '--resolution', type=int, default=7)

    # Taichung, Regensburg, Xiamen, Shenzhen, Ürümqi, Berchtesgaden, Beijing, Shanghai (sea, mountains, desert, city)
    coords = [(24.1, 120.7), (49.0, 12.0), (24.5, 118.1), (22.5, 114.1), (43.8, 87.6), (47.6, 13.0), (39.91, 116.40),
              (31.22, 121.46)]

    args = parser.parse_args()
    model_ps = args.model_path
    p_dir = args.map_path
    user_name = args.user
    password = args.password
    # r = float("{:06.0f}".format(h3_radius(args.resolution)))
    #
    # for coord in coords:
    #     box_pos = box(*Dwn.create_coordinate(coord, -r, -r), *Dwn.create_coordinate(coord, r, r))
    #     zip_files = Dwn.download_best([box_pos], p_dir, user_name, password)
    #
    # for raw_map_path in Dwn.unzip_all_maps(p_dir):
    #     scn, o_path = Dwn.create_image(raw_map_path)
    #
    #     coord_map = SatelliteMap(scn, raw_map_path)
    #
    #     coord_map.classify(args.resolution, {0: "black", 1: "cloud", 2: "rural", 3: "urban", 4: "water"})

    #    # overlay = coord_map.create_raster_image(os.path.join(raw_map_path, "control.tif"))  # join(raw_map_path, "control.tif")

    #    # blended = Image.blend(img, overlay, 0.15)
    #    # blended.save(os.path.join(raw_map_path, "test.tif"))

    geo_file = os.path.join(p_dir, "{}_cls.geojson".format("Taiwan"))

    if not os.path.exists(geo_file):

        taiwan = NationalMap("Taiwan", (user_name, password))

        taiwan.add_local_maps(os.path.join(p_dir, str(taiwan)))
        taiwan.classify_all({0: "black", 1: "cloud", 2: "rural", 3: "urban", 4: "water"})
        tai = taiwan.export_classification(geo_file)

    else:
        tai = geopandas.GeoDataFrame.from_file(geo_file)

    paint_classification(tai, "Taiwan", os.path.join(p_dir, "Taiwan.png"))
    print("Finished")
