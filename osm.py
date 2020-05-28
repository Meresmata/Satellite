import argparse
import math
import os

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from h3_resolution import h3_res


def osm_export(point, radius):
    if -90.0 < point[0] > 90.0:
        raise AttributeError
    if -180.0 < point[1] > 180.0:
        raise AttributeError

    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query_count = """
    [out:json];
    (node(around:{}, {}, {});
    );
    out count;
    """.format(radius, *point)

    overpass_query_count_building = """
    [out:json];
    (node["building"] (around:{}, {}, {});
    );
    out count;
    """.format(radius, *point)

    overpass_query_length_highway = """
    [out:json];
    (way['highway'] (around:{}, {}, {});
    );
    make stat length=sum(length());
    out;
    """.format(radius, *point)

    area = math.pi * radius * radius

    # http requests with retries (10x)
    retry_strategy = Retry(total=4, status_forcelist=[429, 500, 502, 503, 504])

    adapter = HTTPAdapter(max_retries=retry_strategy)
    http = requests.Session()
    http.mount("https://", adapter)
    http.mount("http://", adapter)

    req = lambda x: http.get(overpass_url, params={"data": x}).json()['elements'][0]['tags']

    count = int(req(overpass_query_count)['nodes'])
    building_count = int(req(overpass_query_count_building)['nodes'])
    highway_length = float(req(overpass_query_length_highway)['length'])

    return count / area, building_count / area, highway_length / area


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('main_path', type=str, help='Path to the train main folder of files.')
    parser.add_argument('-r', '--resolution', type=int, default=7)

    args = parser.parse_args()
    p_dir = args.main_path
    res = args.resolution

    center = (24.148, 120.674)
    dist = 12500.0
    test = osm_export(center, dist)

    data_frames = []
    for folder, _, files in os.walk(p_dir):
        csvs = [f for f in files if f.endswith(".csv")]

        for c in csvs:
            data_frames.append(pd.read_csv(os.path.join(folder, c)))

    concat_frame = pd.concat(data_frames)
    radius_m = h3_res[res] * 1000.0
    osm_counts, building_counts, highway_lengths = [osm_export(t, radius_m) for t in
                                                    list(concat_frame['Latitude', 'Longitude'])]

    concat_frame['Osm_Counts'] = osm_counts
    concat_frame['Building_Counts'] = building_counts
    concat_frame['Highway_Lengths'] = highway_lengths
