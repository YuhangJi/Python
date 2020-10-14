import h5py
import os
from platform import system


def getDataFiles(list_filename):
    return [line.strip() for line in open(list_filename,'r')]


def loadH5(h5_filename):
    f = h5py.File(h5_filename,'r')
    data = f['data'][:]
    label = f['label'][:]
    return data, label


def loadDataFile(h5_filename):
    return loadH5(h5_filename)


def getPath(path1,path2):
    return os.path.join(path1,path2)


def match_postfix(postfix, path=None, iscwd=False):
    """match all pointed files in the directory
    :param path: root path
    :param postfix: postfix without “.”
    :param iscwd:
    :return: list
    """
    if iscwd:
        path = os.getcwd()
    return [
        os.path.join(roots,i)
        for roots, dirs, files in os.walk(path, followlinks=False) for i in files if(i.split('.')[-1] == postfix)
    ]


def removeFiles(file_list):
    for i in file_list:
        os.remove(i)


def getSystemOS():
    """
    acquire system OS
    :return: windows:0 linux:1
    """
    c = -1
    my_os = system()
    if my_os == 'Windows': c = 0
    if my_os == 'Linux': c = 1
    return c


def getPathSeparator():
    return "/" if getSystemOS() else "\\"


def getColorDict():
    return {
        0: [255, 215, 0], 1: [178, 34, 34], 2: [0, 191, 255],
        3: [238, 0, 238], 4: [28, 120, 135], 5: [193, 205, 205],
        6: [139, 137, 137]
    }


def getOriginalCoordinates(name,file_path):
    file_list = match_postfix('h5',file_path)
    h5_coordinates = []
    h5_labels = []
    for file_ in file_list:
        if name in file_:
            with h5py.File(file_,'r') as h5_fo:
                h5_coordinates = h5_fo['data'][:]
                h5_labels = h5_fo['label'][:]
            break
    return h5_coordinates,h5_labels


def getDistribution(path,classes):
    """
    统计点云类别分布
    :param path: 点云根目录
    :param classes: 类别的数量
    :return: 字典，｛文件名称1：[类别1数量，类别2数量，...]，文件名称2：[...]，...｝
    """
    file_list = match_postfix('txt',path)
    name_list = [_.split(getPathSeparator())[-1][:-4] for _ in file_list]
    stat_dict = {name_list[_]:[0 for __ in range(classes)] for _ in range(len(name_list))}
    for file in file_list:
        with open(file,'r') as fi:
            lines = [_.strip() for _ in fi.readlines()]
            number_list = [int(_.split(" ")[-1]) for _ in lines]
            file_name = [_ for _ in name_list if _ in file][0]
            for label_number in number_list:
                stat_dict[file_name][label_number] += 1
    return stat_dict

