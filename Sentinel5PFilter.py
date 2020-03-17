import argparse
import concurrent.futures as cf
import os

import geopandas
import pandas as pd


def filter_by_nation(df: geopandas.GeoDataFrame, nation: str) -> geopandas.GeoDataFrame:
    """
    filter of a Geoframe to values in a given Country
    :param df: geopandas.GeoDataFrame
    :param nation: str
    :return: geopandas.GeoDataFrame
    """
    _world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    test_str = 'name=="{}"'.format(nation)
    nation = _world.query(test_str)

    _inner = geopandas.sjoin(df, nation, op="within")

    _geometry = geopandas.points_from_xy(_inner.longitude, _inner.latitude)
    return geopandas.GeoDataFrame(_inner, geometry=_geometry)


def filter_by_percentile(df: geopandas.GeoDataFrame, lower: int = 1, upper: int = 99) -> geopandas.GeoDataFrame:
    """
    Filter Geoframe to exclude Minimum and Maximum values (in percent)
    Not tested
    :param df: geopandas.GeoDataFrame
    :param lower: int
    :param upper: int
    :return: geopandas.GeoDataFrame
    """
    assert lower < upper

    mini = df.value.min
    diff = df.value.max - mini
    min_allowed = mini + lower / 100 * diff
    max_allowed = mini + upper / 100 * diff

    df = df[df.value >= min_allowed]
    df = df[df.value <= max_allowed]

    return df


def cut_country(_path: str) -> None:
    """
    Filter a Dataframe (of a given pkl file path) with the country name (in the path) and save as pkl file
    :param _path: str
    :return: None
    """

    data = pd.read_pickle(_path)

    # Create geopandas dataframe

    geometry = geopandas.points_from_xy(data.longitude, data.latitude)
    data = geopandas.GeoDataFrame(data, geometry=geometry, crs={'init': 'epsg:4326'})

    file_list = _path.split(os.path.sep)[1:-1]
    nation_name = ""
    for name in file_list:  # remove root from path
        if name not in p_dir:
            nation_name = name.capitalize().split("_")[0]
            break
    _path = os.path.split(_path)[0]

    inner = filter_by_nation(data, nation_name)
    inner_name = os.path.join(_path, "inner.pkl")
    inner.to_pickle(inner_name)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='Path to the main folder of files.')
    args = parser.parse_args()

    p_dir = args.path
    pkl_files = []
    for path, _, files in os.walk(p_dir):

        if any([file.endswith(".pkl") for file in files]):
            inner_file = os.path.join(path, "inner.pkl")
            if not os.path.exists(inner_file):
                pkl_files.extend([os.path.join(path, file) for file in files if file.endswith(".pkl")])

    with cf.ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
        _ = list(executor.map(cut_country, pkl_files))
