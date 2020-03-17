import argparse
import concurrent.futures as cf
import os
import typing as tp
from datetime import datetime, date

import geopandas
import matplotlib.pyplot as plt
import pandas as pd
from dateutil.parser import parse
from numpy import average


def parse_path(_path: str) -> tp.Tuple:
    """
    Parse a path to extract the country name, date and sensed gas/spectral line
    :param _path:str
    :return: Tuple of country name, date, sensed gas
    """
    file_list = _path.split(os.path.sep)[1:-1]
    index = 1
    for name in file_list:
        try:
            parse(name)  # remove root from path
            break
        except ValueError:
            index = index + 1

    _nation_name = _path.split(os.path.sep)[index - 1].capitalize()
    _date = _path.split(os.path.sep)[index]
    _gas = [string for string in _path.split(os.path.sep)[index + 1].split("_") if len(string)][1]

    return _nation_name, _date, _gas


def plot_nation_data(_path: str) -> None:
    """
    plot the heat map of the spectrum of a given country (by path) as png in the same path
    :param _path: str
    :return: None
    """

    df = pd.read_pickle(_path)

    nation_name, date, _gas = parse_path(_path)

    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    _nation = world.query('name=="{}"'.format(nation_name))

    try:
        _, miny, _, maxy = df.geometry.total_bounds
    except IndexError:
        print("IndexError: {}".format(_path))
        return
    length = int(maxy - miny)
    # Define base of the plot.
    fig, ax = plt.subplots(1, 1, figsize=(40, 40), dpi=100)

    # Disable the axes
    ax.set_axis_off()

    # Plot the data
    df.plot(
        column='value',  # Column defining the color
        cmap='jet',  # Colormap
        marker='H',  # marker layout. Here a Hexagon.
        markersize=1000 // (length * 3 + 1),
        ax=ax  # Base
    )
    ax.set_title('{} {} Concentration {}'.format(nation_name, _gas, date), fontsize=65)

    # Plot the boundary of the countries on top
    _nation.geometry.boundary.plot(color=None, edgecolor='black', ax=ax)

    plt.savefig(os.path.join(os.path.split(_path)[0], "{}.png".format(nation_name)))
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='Path to the main folder of files.')
    args = parser.parse_args()
    p_dir = args.path

    paths = []
    for path, _, files in os.walk(p_dir):

        if any([file.endswith("inner.pkl") for file in files]):

            file = [file for file in files if file.endswith("inner.pkl")][0]
            f = os.path.join(path, file)
            paths.append(f)

    # plot heat maps
    with cf.ProcessPoolExecutor(max_workers=os.cpu_count()//2) as executor:
        hel = list(executor.map(plot_nation_data, paths))

    # plot average concentration vs. time (CW)
    d = {}
    for x in paths:
        if len(d) == 0:
            d = {parse_path(x)[0]: {parse_path(x)[1]: {parse_path(x)[2]: average(pd.read_pickle(x).value)}}}
        else:
            if parse_path(x)[0] not in list(d.keys()):  # Nation
                d[parse_path(x)[0]] = {parse_path(x)[1]: {parse_path(x)[2]: average(pd.read_pickle(x).value)}}
            else:
                if parse_path(x)[1] not in list(d[parse_path(x)[0]].keys()):  # Date
                    d[parse_path(x)[0]][parse_path(x)[1]] = {parse_path(x)[2]: average(pd.read_pickle(x).value)}
                else:
                    d[parse_path(x)[0]][parse_path(x)[1]][parse_path(x)[2]] = average(pd.read_pickle(x).value)

    for nation in d.keys():
        for gas in list(list(d[nation].values())[0].keys()):
            dates = list(k for k, v in d[nation].items() if gas in v.keys())
            x_values = [datetime.strptime(d, "%Y-%m-%d").date() for d in dates]

            delimiter = date(2020, 1, 1)
            x1 = [date.isocalendar()[1] for date in x_values if date < delimiter]
            x2 = [date.isocalendar()[1] for date in x_values if date >= delimiter]

            values1: tp.List[float] = [v[gas] for k, v in d[nation].items()
                                       if datetime.strptime(k, "%Y-%m-%d").date() < delimiter and gas in v.keys()]
            values2: tp.List[float] = [v[gas] for k, v in d[nation].items()
                                       if datetime.strptime(k, "%Y-%m-%d").date() >= delimiter and gas in v.keys()]

            plt.title('{}: average {} Concentration'.format(nation, gas), fontsize=20)

            plt.plot(x1, values1, label='2019')
            plt.xlabel('calendar week')
            plt.ylabel("concentration")

            plt.plot(x2, values2, label='2020')
            plt.legend()
            plt.ylim([0, None])
            plt.savefig(os.path.join(p_dir, nation, gas))
            plt.close()
