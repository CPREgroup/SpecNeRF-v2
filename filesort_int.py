import glob
import os


def sort_file_int(path, ext):
    imgpaths = sorted(glob.glob(os.path.join(path)),
                      key=lambda x: int(x.split('_')[-1][:-1 - len(ext)]))
    return imgpaths
