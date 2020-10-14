import h5py
import os


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
