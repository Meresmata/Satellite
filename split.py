import argparse
import os
import shutil
from random import randint

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', type=str, help='Path to the input main folder of files.')
    parser.add_argument('out_path', type=str, help='Path to the output main folder of files.')
    parser.add_argument('train_percentage', type=int, help='Percentage of the files to split into training set.')
    args = parser.parse_args()

    in_p = args.in_path
    out_p = args.out_path
    tr_percent = args.train_percentage

    for path, _, files in os.walk(in_p):
        cls = path.split(os.sep)[-1]
        train_folder = os.path.join(out_p, "train", cls)
        test_folder = os.path.join(out_p, "test", cls)

        if files:

            if not os.path.exists(train_folder):
                os.makedirs(train_folder)
            if not os.path.exists(test_folder):
                os.makedirs(test_folder)

            for file in files:
                if randint(0, 100) < tr_percent:
                    shutil.copyfile(os.path.join(path, file), os.path.join(train_folder, file))
                else:
                    shutil.copyfile(os.path.join(path, file), os.path.join(test_folder, file))
