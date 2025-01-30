import imp
import os.path as osp
import os.path

def load(name):
    pathname = osp.join(osp.dirname(__file__), name)
    return imp.load_source('', pathname)
