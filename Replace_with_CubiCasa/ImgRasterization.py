import cairosvg
import numpy as np
import os

from PIL import Image
from numpy import genfromtxt
saveName = 'svgImg_roughcast.png'
svgName = 'model_roughcast.svg'

def convert(imgPth):
    try:
        cairosvg.svg2png(url=os.path.join(imgPth, svgName), write_to=os.path.join(imgPth, saveName))
    except TypeError:
        print('type' + imgPth)
    except ValueError:
        print('value' + imgPth)


if __name__ == "__main__":
    """
    将svg文件转化成RGBA格式的图像
    """
    excludeList = [
        '\\high_quality_architectural\\2003\\',
        '\\high_quality_architectural\\2565\\',
        '\\high_quality_architectural\\6143\\',
        '\\high_quality_architectural\\10074\\',
        '\\high_quality_architectural\\10754\\',
        '\\high_quality_architectural\\10769\\',
        '\\high_quality_architectural\\14611\\',
        '\\high_quality\\7092\\',
        '\\high_quality\\1692\\',
        'high_quality_architectural\\10',  # img does not match label
    ]

    dataDir = "E:\\code\\floor_data\\cubicasa5k"
    dataFile = "\\train.txt"
    all_folders = genfromtxt(dataDir + dataFile, dtype='str')
    folders = []
    for x in all_folders:
        x = x.replace('/', '\\')
        if x not in excludeList:
            folders.append(x)

    fs = [dataDir + x for x in folders]


    import multiprocessing
    import json
    from functools import partial
    from multiprocessing import Pool
    from tqdm import tqdm

    cpuCount = multiprocessing.cpu_count() - 3
    pool = Pool(cpuCount)
    for res in tqdm(pool.imap_unordered(convert, fs), total=len(folders)):  # 将刚才的 fs 列表传进去，让每一个目录都调用 convert 函数
        pass
