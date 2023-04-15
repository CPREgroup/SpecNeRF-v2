import argparse
import glob
import os
import shutil

from filesort_int import sort_file_int

jpgmode = False

parser = argparse.ArgumentParser()
parser.add_argument('--scene_dir', type=str, default='data\spec_data\\test', help='scene directory')
parser.add_argument('--filter_dir', type=str, default='../filters/', help='filters directory')
parser.add_argument('--filter_num', type=int, default=21, help='filter number')
parser.add_argument('--legacy', type=int, help='splitting based on filters (all view points in one filter folder)')
parser.add_argument('--angle_num', type=int, default=16, help='shooting angle number')
parser.add_argument('--img_ext', type=str, default='jpg' if jpgmode else 'dng', help='ext. of img files in scene')

args = parser.parse_args()


print('='*5, 'img files should be the format of xxx_234.ext')


allimg_dir = os.path.join(args.scene_dir, 'jpegs/' if jpgmode else 'RAW/')
filter_num = args.filter_num
angle_num = args.angle_num
legacy = args.legacy

imgpaths = sort_file_int(f'{allimg_dir}/*.{args.img_ext}', args.img_ext)
assert filter_num * angle_num == len(imgpaths), 'numbers dont match!'

def movefile(imgsname, scene_dir):
    for im in imgsname:
        # shutil.move(im, pose_dir)
        _, imname = os.path.split(im)
        print(f'{scene_dir}/{imname}')
        os.symlink(im, f'{scene_dir}/{imname}')


def filter_base():
    for i in range(0, filter_num):
        imgs = imgpaths[i::filter_num]
        print('='*5, i)
        print(imgs)
    
        filter_dir = os.path.join(args.scene_dir, f'filter{i}img_jpegs/images/' if jpgmode else f'filter{i}img/images/')
        if not os.path.exists(filter_dir):
            os.makedirs(filter_dir)
    
        movefile(imgs, filter_dir)
    
        shutil.copy(os.path.join(args.scene_dir, args.filter_dir, f'./f_{i}.mat'), os.path.join(filter_dir, '../'))
        print('done!')
        
        if jpgmode:
            break

def pose_base():
    for i in range(0, angle_num):
        imgs = imgpaths[i * filter_num: (i+1) * filter_num]
        print('=' * 5, i)
        print(imgs)

        pose_dir = os.path.join(args.scene_dir, f'pose{i}img/images/')
        if not os.path.exists(pose_dir):
            os.makedirs(pose_dir)

        movefile(imgs, pose_dir)

        print('done!')


if __name__ == '__main__':
    if legacy:
        filter_base()
    else:
        pose_base()

