import argparse
import concurrent.futures as cf
import os
from datetime import datetime, timedelta

import certifi
import geopandas
import pandas as pd
import s5a
import sentinel5dl


def monday_of_calender_week(_year: int, _week: int) -> datetime:
    """
    calculates the first day of the week, of a given year and calendar week
    :param _year: int
    :param _week: int
    :return: datetime
    """
    first = datetime(_year, 1, 1)
    base = 1 if first.isocalendar()[1] == 1 else 8
    return first + timedelta(days=base - first.isocalendar()[2] + 7 * (_week - 1))


def download_sentinel5_offline(start_date: datetime, length: timedelta, _country_name: str = None, _path: str = ".",
                               product: str = None) -> None:
    """
    Download the satellite data of a given start date for a given length. Filtered by Country Name,
    :param start_date: datetime
    :param length: timedelta
    :param _country_name: str
    :param _path: str
    :param product: str name of the Gas, Spectral Region
    :return: None
    """
    begin_date = '{}.000Z'.format(start_date.isoformat())
    end_date = '{}.999Z'.format((start_date + length).isoformat())

    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

    _level = None
    if product is not None:

        if product in ["IR_SIR", "IR_UVN", "RA_BD1", "RA_BD2", "RA_BD3", "RA_BD4", "RA_BD5", "RA_BD6", "RA_BD6",
                       "RA_BD7", "RA_BD8"]:
            product = "LIB_{}".format(product)
            _level = "L1B"

        if product in ["AER_AI", "AER_LH", "CH4", "CLOUD", "CO", "HCHO", "NP_BD3", "NO2", "NP_BD6", "NP_BD7", "O3_TCL",
                       "O3", "SO2"]:
            product = "L2__{}_____".format(product)[0:10]
            _level = "L2"

    if _country_name is not None:
        nation = world.query('name=="{}"'.format(_country_name))
        minx, miny, maxx, maxy = nation.geometry.total_bounds
        _country = "POLYGON(({0:.2f} {1:.2f},{0:.2f} {3:.2f},{2:.2f} {3:.2f},{2:.2f} {1:.2f},{0:.2f} {1:.2f}))".format(
            minx, miny, maxx, maxy)
    else:
        _country = None

    # Search for Sentinel-5 products
    result = sentinel5dl.search(
        polygon=_country,
        begin_ts=begin_date,
        end_ts=end_date,
        product=product,
        processing_level=_level,
        processing_mode='Offline')

    if not any([value is None for value in [_country_name, start_date, product]]):
        _path = os.path.join(_path, _country_name, start_date.strftime("%Y-%m-%d"), product)

    if not os.path.exists(_path):
        os.makedirs(_path)

    # Download found products to the local folder
    sentinel5dl.download(products=result.get("products"), output_dir=_path)


def download_sentinel5_cw(start_year: int, cw: int, _filter: str = None, _country: str = None, _path: str = ".",
                          product: str = None) -> None:
    """
    Download the satellite data of Sentinel5 for a given year and calendar week. filtered my different categories
    of days and a given country
    :param start_year: int
    :param cw: int
    :param _filter: str
    :param _country: str
    :param _path: str
    :param product: str name of the Gas, Spectral Region
    :return: None
    """
    _filter = _filter.lower()
    if _filter not in ["weekend", "weekday", "mwf", "tt", "mon", "tue", "wed", "thu", "fri", "sat", "sun", None]:
        raise AttributeError

    if cw > 53 or cw <= 0:
        raise AttributeError

    dates = []
    if _filter == "weekend":
        dates = [5, 6]
    elif _filter is None:
        dates = range(0, 7)
    elif _filter == "weekday":
        dates = range(0, 5)
    elif _filter == "mwf":
        dates = [0, 2, 4]
    elif _filter == "tt":
        dates = [1, 3]
    elif _filter == "mon":
        dates = [0]
    elif _filter == "tue":
        dates = [1]
    elif _filter == "wed":
        dates = [2]
    elif _filter == "thu":
        dates = [3]
    elif _filter == "fri":
        dates = [4]
    elif _filter == "sat":
        dates = [5]
    elif _filter == "sun":
        dates = [6]

    week_date = monday_of_calender_week(start_year, cw)
    delta = timedelta(days=0, hours=23, minutes=59, seconds=59)
    for date in dates:
        start_date = week_date + timedelta(days=date)
        download_sentinel5_offline(start_date, delta, _country, _path, product)


def to_pickle(directory: str) -> None:
    """
    save the data as pickled file (.pkl)  after filtering the values to with h3 to a resolution of 6 (3.229482772 km
    hexagonal length)
    :param directory: str
    :return: None
    """
    _files = [file for file in os.listdir(directory) if file.endswith(".nc")]

    data = []

    for file in _files:
        f = os.path.join(directory, file)
        try:
            data.append(s5a.load_ncfile(f))
        except OSError:
            print("OSERROR: {}".format(f))

    data = pd.concat(data, ignore_index=True)

    data = s5a.point_to_h3(data, resolution=6)
    data = s5a.aggregate_h3(data)
    data = s5a.h3_to_point(data)

    data.to_pickle(path=os.path.join(directory, "data.pkl"))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='Path to the main folder of files.')
    parser.add_argument('-g', '--gas', type=str)
    parser.add_argument('-c', '--countries', type=str, action='append',
                        help='Countries to filter', default=None)
    args = parser.parse_args()

    p_dir = args.path
    countries = args.countries
    gas = args.gas

    # certifi needed on windows
    certifi.where()
    sentinel5dl.ca_info = certifi.where()

    for country in countries:
        for year in [2019, 2020]:
            for week in range(1, 9):
                # download_sentinel5_cw(_year, week, _country=country, _path=p_dir, product=gas, _filter="wed")
                pass

    # does not with with h3 on windows?
    nc_path = []
    for path, _, files in os.walk(p_dir):
        nc_path.extend([path for file in files if file.endswith(".nc")])
    nc_path = set(nc_path)

    with cf.ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
        list(executor.map(to_pickle, nc_path))

    for path, _, files in os.walk(p_dir):

        for file in files:
            if file.endswith(".nc"):
                os.remove(os.path.join(path, file))
