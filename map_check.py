import argparse
import os
import typing as tp
import warnings

import cupy
import numpy as np
import pandas as pd
import pyproj
from PIL import Image, ImageDraw
from satpy import scene
from shapely.geometry import box

import SentinelSatDownload as Dwn
import classificator as clf
from h3_resolution import h3_res


class Patch:
    """
    represents a patch of a satellite map of a given size, that holds a distinct classifier
    for the raster analysis of the map
    """

    def __init__(self, _img: Image.Image, pixels: tp.Tuple[int, int, int, int], _coords: tp.Tuple[int, int, int, int],
                 map_name: str, map_system: str):
        """
        making the patch somewhat smaller to create a check board looking overlay
        :param pixels:tp.Tuple[int, int, int, int], (left, upper, right, lower) of pixels
        :param _coords: tp.Tuple[int, int, int, int]
        :param map_name: str, name of the parent
        :param map_system: str, epsg, code
        """
        self.p_minx = pixels[0] + 4
        self.p_maxx = pixels[2] - 4
        self.p_miny = pixels[1] + 4
        self.p_maxy = pixels[3] - 4
        self.img = _img
        self.coords = _coords
        self.center = box(*_coords).centroid.coords
        self.system = map_system

        self._classifier: tp.Optional[str] = None
        self.name = str("RGB_{}".format([int(c) for c in _coords]))
        self.map_name: str = map_name

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


class SatelliteMap:
    """
    A given satellite map of a given size, with the range of of  coordinates in a distinct coordinate system
    """

    def __init__(self, _img: Image.Image, _scn: scene, shape=None):
        """
        :param _img: Image
        :param _scn: scene, of the original satellite map
        .:param shape: shape
        """
        area = _scn['true_color'].attrs['area']
        self.__minx = area.area_extent[0]
        self.__maxx = area.area_extent[2]
        self.__miny = area.area_extent[1]
        self.__maxy = area.area_extent[3]
        self.map = _img
        self.scn = _scn

        self.system: str = area.crs.srs  # epsg-code
        self.shape = box(*area.area_extent) if shape is None else shape.intersection(box(*area.area_extent))
        self.__p_maxx = area.height
        self.__p_maxy = area.width
        self.patches: tp.Dict[str, Patch] = {}

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
        for _patch in self.patches.values():
            _cls = _patch.get_classifier()
            if _cls == "urban":
                _img[_patch.p_miny:_patch.p_maxy, _patch.p_minx:_patch.p_maxx] = np.array([255, 0, 0])
            elif _cls == "rural":
                _img[_patch.p_miny:_patch.p_maxy, _patch.p_minx:_patch.p_maxx] = np.array([0, 255, 0])
            elif _cls == "outer":
                _img[_patch.p_miny:_patch.p_maxy, _patch.p_minx:_patch.p_maxx] = np.array([0, 255, 255])
            elif _cls == "mixed":
                _img[_patch.p_miny:_patch.p_maxy, _patch.p_minx:_patch.p_maxx] = np.array([255, 215, 0])
            elif _cls == "error":
                _img[_patch.p_miny:_patch.p_maxy, _patch.p_minx:_patch.p_maxx] = np.array([0, 0, 0])

        _img = Image.fromarray(_img)

        # add the coordinates of the patch in the image as text
        d = ImageDraw.Draw(_img)
        for _patch in self.patches.values():
            d.text((_patch.p_minx, _patch.p_miny), str(_patch.coords), fill=(0, 0, 255))

        if image_path:
            _img.save(image_path)
        return _img

    def classify(self, resolution: int, cls_dict: tp.Dict[int, str]):

        _r = float("{:04.2f}".format(h3_res[resolution] * 0.866))
        pixel_boxes, coord_boxes = Dwn.create_xy_bbox(self.scn, 2 * _r * 1000)
        tmp = []
        for p_box, c_box in zip(pixel_boxes, coord_boxes):
            _img, _ = Dwn.crop_image_by_points(self.map, p_box, c_box)  # raw_map_path

            patch = Patch(_img, p_box, c_box, str(self), self.system)

            # noinspection PyTypeChecker
            if is_containing_black(cupy.asarray(_img)):
                # classify as black / error
                patch.set_classifier("error")
                self.add_patch(patch)
            elif not self.shape.contains(box(*c_box)):
                # classify as outer / sea
                patch.set_classifier("outer")
                self.add_patch(patch)
            else:
                tmp.append(patch)

        imgs = [p.img for p in tmp]
        classifier = clf.get_classifier(imgs, model_ps[0], cls_dict)

        temp_dict = {str(v.set_classifier(classifier[i])): v.set_classifier(classifier[i]) for i, v in enumerate(tmp)}

        self.patches.update(temp_dict)


class NationalMap:

    def __init__(self, _name: str, user_credentials: tp.Tuple[str, str], resolution: int = 7):
        self.__box = Dwn.box_of_nation(_name)
        self.__shape = Dwn.shape_of_nation(_name)
        self.__maps: tp.Dict[str, SatelliteMap] = {}
        self.__usr_name = user_credentials[0]
        self.__pw = user_credentials[1]
        self.name = _name
        self.resolution = float("{:04.2f}".format(h3_res[resolution] * 0.866))

    def __str__(self):
        return self.name

    def add_maps(self, folder: str, _box: tp.Optional[tp.Tuple] = None):
        _box = _box if type(_box) is not None else self.__box
        _zip_files = Dwn.download_best(_box, folder, self.__usr_name, self.__pw)

        for _ in Dwn.unzip_maps(_zip_files):
            _scn, path = Dwn.create_image()
            _img = Image.open(path)

            sat_map = SatelliteMap(_img, _scn['true_color'].attrs['area'])

            self.__maps[str(sat_map)] = sat_map

    def classify_all(self, cls_dict: tp.Dict):

        tmp: tp.List[Patch] = []
        for name, _map in self.__maps.items():
            pixel_boxes, coord_boxes = Dwn.create_xy_bbox(_map.scn, 2 * self.resolution * 1000)

            for p_box, c_box in zip(pixel_boxes, coord_boxes):
                _img, _ = Dwn.crop_image_by_points(_map.map, p_box, c_box)  # raw_map_path

                patch = Patch(_img, p_box, c_box, name, _map.system)

                # noinspection PyTypeChecker
                if is_containing_black(cupy.asarray(_img)):
                    # classify as black / error
                    patch.set_classifier("error")
                    _map.add_patch(patch)
                elif not _map.shape.contains(coord_boxes):
                    # classify as outer / sea
                    patch.set_classifier("outer")
                    _map.add_patch(patch)
                else:
                    tmp.append(patch)

        imgs = [p.img for p in tmp]
        classifier = clf.get_classifier(imgs, model_ps[0], cls_dict)

        tmp: tp.List[Patch] = [v.set_classifier(classifier[i]) for i, v in enumerate(tmp)]

        for temp_elem in tmp:
            self.__maps[temp_elem.map_name].patches[str(temp_elem)] = temp_elem

    def export_classification(self, path: str):
        exportable: tp.List[Patch] = []

        for v_maps in self.__maps.values():
            for v_patch in v_maps.patches.values():
                if any([v_patch.center - elem.center > self.resolution for elem in exportable]):
                    exportable.append(v_patch)
                    #  TODO test for older patch classification, set mixed or too new one if older was black

        # convert to EPSG 4326
        longitudes, latitudes, classifications = [], [], []
        out_proj = pyproj.Proj(init="epsg:4326")

        for e in exportable:
            in_proj = pyproj.Proj(e.system)
            lon, lat = pyproj.transform(in_proj, out_proj, *e.center)
            longitudes.append(lon), latitudes.append(lat), classifications.append(e.get_classifier())

        df = pd.DataFrame(list(zip(longitudes, latitudes, classifications)))
        df.to_pickle(path)


def is_containing_black(np_img: cupy.ndarray) -> bool:
    # whether image black
    black = cupy.array([0, 0, 0])
    is_pixel_black = cupy.all(np_img == black, axis=-1)
    black_count = len(is_pixel_black[is_pixel_black])

    # accept 85 % black pixels, #(pixels) = image_size // 3 (number of channels)
    return black_count > 0.15 * np_img.size // 3


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
    r = float("{:04.2f}".format(h3_res[args.resolution] * 0.866))

    # for coord in coords:
    #     box_pos = box(*Dwn.create_coordinate(coord, -r, -r), *Dwn.create_coordinate(coord, r, r))
    #     zip_files = Dwn.download_best(box_pos, p_dir, user_name, password)
    #
    # for raw_map_path in Dwn.unzip_all_maps(p_dir):
    #     scn, o_path = Dwn.create_image(raw_map_path)
    #
    #     img = Image.open(o_path)
    #     coord_map = SatelliteMap(img, scn)
    #
    #     coord_map.classify(args.resolution, {0: "rural", 1: "urban"})
    #
    #     overlay = coord_map.create_raster_image()  # join(raw_map_path, "control.tif")
    #
    #     #  overlayed = Image.blend(img, overlay, 0.15)
    #     #  overlayed.save(os.path.join(raw_map_path, "test.tif"))

    taiwan = NationalMap("Taiwan", (user_name, password))

    # TODO testing
    taiwan.add_maps(os.path.join(p_dir, str(taiwan)))
    taiwan.classify_all({0: "rural", 1: "urban"})
    taiwan.export_classification(os.path.join(p_dir, "{}_cls.pkl".format(str(taiwan))))
