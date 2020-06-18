import argparse
import math
import os

import geopandas
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from shapely.geometry import Polygon, MultiPolygon, box, shape

from h3_resolution import h3_radius


def __osm_request(query: str):
    overpass_url = "http://overpass-api.de/api/interpreter"

    # http requests with retries (10x)
    retry_strategy = Retry(total=4, status_forcelist=[429, 500, 502, 503, 504])

    adapter = HTTPAdapter(max_retries=retry_strategy)
    http = requests.Session()
    http.mount("https://", adapter)
    http.mount("http://", adapter)

    req = lambda x: http.get(overpass_url, params={"data": x}).json()

    return req(query)


def get_osm_national_box(name: str) -> Polygon:
    return box(get_osm_national_boundary(name)['bounds'].values())


def get_osm_national_shapes(name: str) -> Polygon:
    shapes = []
    for member in get_osm_national_boundary(name)['members']:
        if 'geometry' in member:
            shapes.append(shape(member['geometry']))

    return MultiPolygon(shapes)


def get_osm_national_boundary(name: str) -> MultiPolygon:
    overpass_query_shape = """
    [out:json];
    rel
       ["name:en"="{}"]
       ["boundary"="administrative"]
       [admin_level=2];
    out geom qt;
    """.format(name)

    shp = __osm_request(overpass_query_shape)['elements'][0]
    shp = MultiPolygon([Polygon([tuple(i.values()) for i in s['geometry']])
                        for s in shp['members'][2:-1] if s['role'] == 'outer'])
    return shp


def osm_export(point, radius):
    if -90.0 < point[0] > 90.0:
        raise AttributeError
    if -180.0 < point[1] > 180.0:
        raise AttributeError

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

    count = int(__osm_request(overpass_query_count)['elements'][0]['tags']['nodes'])
    building_count = int(__osm_request(overpass_query_count_building)['elements'][0]['tags']['nodes'])
    highway_length = float(__osm_request(overpass_query_length_highway)['elements'][0]['tags']['length'])

    return count / area, building_count / area, highway_length / area


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('main_path', type=str, help='Path to the train main folder of files.')
    parser.add_argument('-r', '--resolution', type=int, default=7)

    args = parser.parse_args()
    p_dir = args.main_path
    res = args.resolution

    s = get_osm_national_boundary("Taiwan")

    center = (24.148, 120.674)
    dist = 12500.0
    test = osm_export(center, dist)

    data_frames = []
    for folder, _, files in os.walk(p_dir):
        jsons = [f for f in files if f.endswith(".geojson")]

        for c in jsons:
            data_frames.append(geopandas.read_file(os.path.join(folder, c)))

    concat_frame = geopandas.GeoDataFrame(pd.concat(data_frames, ignore_index=True))
    radius_m = h3_radius(res)
    osm_counts, building_counts, highway_lengths = [osm_export(t, radius_m) for t in
                                                    list(concat_frame['Latitude', 'Longitude'])]

    concat_frame['Osm_Counts'] = osm_counts
    concat_frame['Building_Counts'] = building_counts
    concat_frame['Highway_Lengths'] = highway_lengths
