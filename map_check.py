import argparse
import typing as tp
import warnings
from os.path import join

import numpy as np
from PIL import Image
from pyresample.geometry import AreaDefinition
from shapely.geometry import box

from SentinelSatDownload import download_best, unzip_maps, crop_image_by_points, create_image, create_xy_bbox, \
    create_coordinate
from classificator import get_classifier
from h3_resolution import h3_res


class Patch:
    """
    represents a patch of a satellite map of a given size, that holds a distinct classifier
    for the raster analysis of the map
    """

    def __init__(self, pixels: tp.Tuple[int, int, int, int]):
        """
        making the patch somewhat smaller to create a checkboard looking overlay
        :param pixels:tp.Tuple[int, int, int, int], (left, upper, right, lower) of pixels
        """
        self.p_minx = pixels[0] + 4
        self.p_maxx = pixels[2] - 4
        self.p_miny = pixels[1] + 4
        self.p_maxy = pixels[3] - 4

        self._classifier: tp.Optional[str] = None

    def set_classifier(self, _classifier: str) -> None:
        """
        setting the _classifier of the of that element of the raster
        :param _classifier: str, classifier from the neural network
        :return: None
        """
        if self._classifier is None or self._classifier == _classifier:
            self._classifier = _classifier
        else:
            self._classifier = "Mixed"

    def set_multi_classifiers(self, _classifiers: tp.List[str]) -> None:
        map(self.set_classifier, _classifiers)

    def get_classifier(self) -> str:
        """
        returns the classifier name
        :return: str, the classifier name
        """
        return self._classifier


class SatelliteMap:
    """
    A given satellite map of a given size, with the range of of  coodinates in a destinct coordinate system
    """

    def __init__(self, area: AreaDefinition):
        """
        :param area: AreaDefinition, of the original satellite map
        """
        self.__minx = area.area_extent[0]
        self.__maxx = area.area_extent[2]
        self.__miny = area.area_extent[1]
        self.__maxy = area.area_extent[3]

        self.system: str = str(area.crs)  # epsg-code

        self.__p_maxx = area.height
        self.__p_maxy = area.width
        self.patches: tp.List = []
        self.system = area

    def add_patch(self, _patch: Patch):
        """
        :param _patch: Patch itself
        :return:
        """
        self.patches.append(_patch)

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
        for _patch in self.patches:
            if _patch.get_classifier() == "urban":
                _img[_patch.p_miny:_patch.p_maxy, _patch.p_minx:_patch.p_maxx] = np.array([255, 0, 0])
            elif _patch.get_classifier() == "rural":
                _img[_patch.p_miny:_patch.p_maxy, _patch.p_minx:_patch.p_maxx] = np.array([0, 255, 0])
            elif _patch.get_classifier() == "mixed":
                _img[_patch.p_miny:_patch.p_maxy, _patch.p_minx:_patch.p_maxx] = np.array([0, 0, 255])

        _img = Image.fromarray(_img)
        if image_path:
            _img.save(image_path)
        return _img


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

    for coord in coords:
        box_pos = box(*create_coordinate(coord, -r, -r), *create_coordinate(coord, r, r))

        download_best(box_pos, p_dir, user_name, password)

    for raw_map_path in unzip_maps(p_dir):
        scn, o_path = create_image(raw_map_path)

        image = Image.open(o_path)
        boxes = create_xy_bbox(scn, 2 * r * 1000, False)

        coord_map = SatelliteMap(scn['true_color'].attrs['area'])

        tmp_boxes, tmp_patches = [], []
        for xy_box in boxes:
            img, name = crop_image_by_points(image, scn['true_color'].attrs['area'], xy_box, raw_map_path)
            tmp_patches.append(img)
            tmp_boxes.append(xy_box)

        classifiers = get_classifier(tmp_patches, model_ps[0], {0: "rural", 1: "urban"})

        for i in range(len(tmp_patches)):
            patch = Patch(tmp_boxes[i])
            patch.set_classifier(classifiers[i])
            coord_map.add_patch(patch)

        overlay = coord_map.create_raster_image()

        overlayed = Image.blend(image, overlay, 0.1)
        overlayed.save(join(raw_map_path, "test.png"))
