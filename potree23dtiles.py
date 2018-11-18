#!coding=utf-8
import numpy as np
from collections import OrderedDict

from .lasio import las_
from .proj import inv_wgs84, trans_wgs84
import glob2
from .pnts import *

_STEP = 20

def read_las(fname, koi=('rgb','class'),tm='49n'):
    '''
    读取点云文件
    :param fname: 文件名
    :param koi: 读取的字段
    :param scale: 比例尺
    :return: 返回数据的dist
    '''
    las = las_(fname)
    ncout = las.count()

    Step = 1 << _STEP
    boundary = []
    xyz = []

    # 判断数据里边是否存在指定的属性
    records = [key[2:] for key in las._RecordTypes[las.get_record_id()].keys()]

    # 初始化字典
    attribute = {}
    _tmpkoi = []
    for k in koi:
        if k not in records:
            continue
        attribute[k] = []
        _tmpkoi.append(k)
    koi = _tmpkoi
    _scale = las.scale

    for i in range(0, ncout, Step):
        arr = las.query(i, i + Step)
        n = arr.shape[0]
        if n == 0:
            continue
        if xyz is not None:
            xyz.append((arr['xyz']).astype('i4'))
        for k in koi:
            attribute[k].append(arr[k])
        boundary.append(xyz[len(xyz)-1].min(0)*_scale)
        boundary.append(xyz[len(xyz)-1].max(0)*_scale)

    # 防止没有颜色信息
    if not attribute.get('rgb',None):
        attribute['rgb']=np.zeros((ncout,3),dtype='u1')

    boundary = np.vstack(boundary)
    boundary = np.vstack([boundary.min(0), boundary.max(0)])
    boundary = (boundary + las.offset).T
    xyz = np.vstack(xyz)

    # 存储数据到数组
    for k in koi:
        if len(attribute[k][0].shape)>1:
            attribute[k] = np.vstack(attribute[k])
        else:
            attribute[k] = np.hstack(attribute[k])

    attribute['treeID'] = np.ones(attribute['class'].shape).astype('u2')
    attribute['rgb']=attribute['rgb'].astype('u1')
    # 构造结果字典
    pcd= {
        'xyz': xyz,
        'attr': attribute,
        'metainfo':{        # 元数据
            'box': boundary,
            'scale':las.scale,
            'count':ncout,
            'offset': las.offset,
            'record_id':las.record_id
        }
    }
    #covert_neu(pcd,tm=tm)
    return pcd


def covert_neu(info,tm='49n',popM=None):
    '''
    转换成neu坐标的数据,将修改info的值
    :param info:读取后的字典值
    :param tm:投影带参数
    :return:no
    '''
    # -- center
    if info is None:
        return None
    _xyz = info.get('xyz')
    _meta = info.get('metainfo')
    _offset = _meta.get('offset')
    _scale = _meta.get('scale')
    if _xyz is not None and _meta is not None:
        if popM is  None:
            mu = _xyz.mean(0) + _offset
            mu = wgs84_from(*(mu), tm=tm)
            popM = trans_wgs84(*mu)  # neu   ->  wgs84
        m = inv_wgs84(popM)  # wgs84 ->  neu
        _xyz = _xyz*_scale+_offset
        _xyz = wgs84_from(*(_xyz.T), tm=tm)
        _xyz = _xyz.dot(m[:3,:3]) + m[3,:3]

        pMax = _xyz.max(0)
        pMin = _xyz.min(0)

        center  = (pMax + pMin)/2
        half = (pMax - pMin) / 2

        bbox = np.r_[center.flatten(), (np.identity(3) * half.max()).flatten()]
        tbbox = np.r_[center.flatten(), (np.identity(3) * half).flatten()]

        #转换成i4,防止精度丢失
        _xyz = (_xyz/_scale).astype('i4')

        # 修改xyz的值
        info['xyz'] = _xyz
        info['neu'] = {
            'bbox':list(bbox.flatten()),
            'scale': _scale,
            'tbbox': list(tbbox.flatten()),
        }
    return info

def pcd2pnts(pcd,outfile):
    xyz = pcd['xyz']
    attr = pcd['attr']
    scale = pcd['neu']['scale']
    xyz = (xyz * scale).astype('f4')
    data = {
        'feature': {
            'POSITION': xyz,
            'RGB': attr['rgb']
        },
        'batch': {
            'class': attr['class'],
            'treeID': attr['treeID']
        },
    }
    feature_data = data.get('feature')
    batch_data = data.get('batch')
    pnts = Pnts()
    pnts.write(outfile, data)

def convertPotree():
    pass




class treeNode:
    def __init__(self):
        self.parent = None
        self.childs = []
        #self.content = {}
        self.key = ''
        self.level = 0
        self.file = ''
        self.hierarchy = False

    def getParent(self):
        return  self.parent

    def addNode(self,node):
        if node.key == self.key:
            if self.parent:
                self.parent.childs.append(node)
                node.parent = self.parent
                print('gen_tree',node.key)
            return
        elif node.level == self.level +1 and node.key[0:-1] == self.key:
            self.childs.append(node)
            node.parent = self
            print('gen_tree',node.key)
            return
        else:
            for e in self.childs:
                e.addNode(node)

    def setFile(self,file):
        self.file = file
        self.key = os.path.basename(file).split('.')[0]
        self.level = len(self.key) - 1
        if self.level%5==0:
            self.hierarchy = True


def visitNode(childs,tileset_json,tm='',popM='',outdir=r'D:\Program Files (x86)\HiServer\apache2.2\htdocs\pcdtest1\potree'):
    if not childs:return

    for e in childs:
        _child_node = {
        'boundingVolume': {'box':[]},  # 存储八叉树的box
        'children': [],
        'content': {'url': ''},  # 存储tightbox , 'boundingVolume': ''
        'geometricError': 0,
        }
        _pcd = read_las(e.file, tm=tm)
        _pcd = covert_neu(_pcd, tm=tm, popM=popM)
        pcd2pnts(_pcd, r'%s/%s.pnts' % (outdir, e.key))
        _child_node['boundingVolume']['box'] = _pcd.get('neu').get('bbox')
        _child_node['geometricError'] = _pcd.get('neu').get('bbox')[3] / geomeotric_space
        _child_node['content']['url'] = '%s.pnts' % (e.key)
        tileset_json.append(_child_node)
        print('visit',e.key)
        if not e.childs:   continue
        visitNode(e.childs,_child_node['children'],tm=tm, popM=popM,outdir=outdir)





class Tree:
    root = treeNode()

    def insert(self,node):
        self.root.addChild(node)


hierarchyStepSize=5

if __name__ == "__main__":
    import os
    import copy
    tm = 'EPSG:32650'
    bbox = {
        'bbox':{
            "lx": 536982.3269807688,
            "ly": 2805172.6864022344,
            "lz": 228.40542941074819,
            "ux": 572545.4316293589,
            "uy": 2840735.7910508245,
            "uz": 35791.510078000836
        },
        'tbbox':{
            "lx": 536982.3269807688,
            "ly": 2805172.6864022344,
            "lz": 228.40542941074819,
            "ux": 543434.6162458079,
            "uy": 2840735.7910883247,
            "uz": 1002.2817169420287
        }
    }
    tightBox = bbox.get('tbbox')
    mu = np.array([(tightBox['lx']+tightBox['ux'])/2,(tightBox['ly']+tightBox['uy'])/2,(tightBox['lz']+tightBox['uz'])/2])

    #记录转换矩阵
    mu = wgs84_from(*(mu), tm=tm)
    popM = popM_from(*mu)  # neu   ->  wgs84
    #m = inv_popM(popM)  # wgs84 ->  neu
    las_list = glob2.glob("%s/**/*.las"%(r'D:\Program Files (x86)\HiServer\apache2.2\htdocs\potree\pointclouds\test\data\r'))
    outdir = r'D:\Program Files (x86)\HiServer\apache2.2\htdocs\pcdtest1\potree1'


    def getkey(name):
        name = os.path.basename(name)
        return len(name)
    # 按数字进行排序
    las_list.sort(key=getkey)

    root = {
        'boundingVolume': {'box': []},  # 存储八叉树的box
        'children': [],
        'content': {'url': ''},  # 存储tightbox , 'boundingVolume': ''
        'geometricError': 0,
        'refine': 'ADD',
        'transform': list(popM.flatten()),  # r4x4, neu 2 wgs84
    }

    geomeotric_space =16
    rootnode = treeNode()

    rootnode.setFile(las_list[0])
    for e in las_list:
        _node = treeNode()
        _node.setFile(e)
        rootnode.addNode(_node)

    pcd = read_las(rootnode.file, tm=tm)
    pcd = covert_neu(pcd, tm=tm, popM=popM)

    pcd2pnts(pcd, r'%s/%s.pnts' % (outdir, rootnode.key))
    root['boundingVolume']['box'] = pcd.get('neu').get('bbox')
    root['geometricError'] = pcd.get('neu').get('bbox')[3] / geomeotric_space
    root['content']['url'] = '%s.pnts' % (rootnode.key)

    visitNode(rootnode.childs,root['children'],tm,popM,outdir)
    tileset = {
        'asset': {'version': '0.0'},
        'geometricError': root['geometricError'],
        'root': root,
    }
    json.dump(tileset, open(r'%s/tileset.json'%outdir, 'w'))

