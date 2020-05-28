import argparse
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str, help='Path to the input main folder of files.')

args = parser.parse_args()
in_p = args.path

# change preliminary naming schemes to final one
for path, _, files in os.walk(in_p):
    for old_file_name in files:
        name_list = re.split(r"[\[\]()]", old_file_name)
        file_ending = old_file_name.split(".")[-1]
        new_file_name = name_list[0] + "[" + name_list[1] + "]." + file_ending

        if new_file_name == old_file_name:
            pass
            # file name is unchanged --> do nothing
        elif os.path.exists(os.path.join(path, new_file_name)):
            #  renamed file already existed --> rem old_file_name
            os.remove(os.path.join(path, old_file_name))
        else:
            #  renamed file name, is still free -> rename to new name scheme
            os.rename(os.path.join(path, old_file_name), os.path.join(path, new_file_name))

print("renamed all!")
