#!coding=utf-8
import numpy as np
from collections import OrderedDict

from lasio import las_
from proj import inv_wgs84, trans_wgs84,wgs84_from,wgs84_trans_matrix
import glob2
from pnts import *

_STEP = 20

def read_las(fname, attr_list=('rgb','class'),tm='49n'):
    '''
    read las file
    :param fname: file name
    :param attr_list:
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

    # init attribute of pointcloud
    attribute = {}
    _tm_attr_list = []
    for k in attr_list:
        if k not in records:
            continue
        attribute[k] = []
        _tm_attr_list.append(k)
    attr_list = _tm_attr_list
    _scale = las.scale

    for i in range(0, ncout, Step):
        arr = las.query(i, i + Step)
        n = arr.shape[0]
        if n == 0:
            continue
        if xyz is not None:
            xyz.append((arr['xyz']).astype('i4'))
        for k in attr_list:
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
    for k in attr_list:
        if len(attribute[k][0].shape)>1:
            attribute[k] = np.vstack(attribute[k])
        else:
            attribute[k] = np.hstack(attribute[k])

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


def covert_neu(info,tm,transM=None):
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
        if transM is  None:
            mu = _xyz.mean(0) + _offset
            mu = wgs84_from(*(mu), tm=tm)
            popM = trans_wgs84(*mu)  # neu   ->  wgs84
        m = inv_wgs84(transM)  # wgs84 ->  neu
        _xyz = _xyz*_scale+_offset
        _xyz = wgs84_from(*(_xyz.T), tm=tm)
        _xyz = _xyz.dot(m[:3,:3]) + m[3,:3]

        pMax = _xyz.max(0)
        pMin = _xyz.min(0)

        center  = (pMax + pMin)/2
        half = (pMax - pMin) / 2

        bbox = np.r_[center.flatten(), (np.identity(3) * half.max()).flatten()]
        tbbox = np.r_[center.flatten(), (np.identity(3) * half).flatten()]

        #convert to i4,in case the loss of precision
        _xyz = (_xyz/_scale).astype('i4')

        # convert xyz to wgs84 neu
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
            'class': attr['class']
        },
    }
    feature_data = data.get('feature')
    batch_data = data.get('batch')
    pnts = Pnts()
    pnts.write(outfile, data)


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

#after potree,the node exist just one point,this situtation can't make the box,
# so if the number of points less than limit_node_size,i just abandon it
limit_node_size=4
def visitNode(childs,tileset_json,tm='',transM=None,outdir=''):
    if not childs:return

    for e in childs:
        _child_node = {
        'boundingVolume': {'box':[]},  # save node box
        'children': [],
        'content': {'url': ''},  #save tightbox , 'boundingVolume': ''
        'geometricError': 0,
        }
        _pcd = read_las(e.file, tm=tm)
        if _pcd['xyz'].shape[0] < limit_node_size: continue
        _pcd = covert_neu(_pcd, tm=tm, transM=transM)
        pcd2pnts(_pcd, r'%s/%s.pnts' % (outdir, e.key))
        _child_node['boundingVolume']['box'] = _pcd.get('neu').get('bbox')
        _child_node['geometricError'] = _pcd.get('neu').get('bbox')[3] / geomeotric_space
        _child_node['content']['url'] = '%s.pnts' % (e.key)
        tileset_json.append(_child_node)
        print('write node:',e.key)
        if not e.childs:   continue
        visitNode(e.childs,_child_node['children'],tm=tm, transM=transM,outdir=outdir)


import os
import copy
def testConvert():
    src = r'D:\Program Files (x86)\HiServer\apache2.2\htdocs\potree\pointclouds\test\data\r'
    # out dir
    outdir = r'D:\Program Files (x86)\HiServer\apache2.2\htdocs\pcdtest1\potree'
    proj_param = 'EPSG:32650'
    # box from cloud.js in potree result
    bbox = {
        'bbox': {
            "lx": 536982.3269807688,
            "ly": 2805172.6864022344,
            "lz": 228.40542941074819,
            "ux": 572545.4316293589,
            "uy": 2840735.7910508245,
            "uz": 35791.510078000836
        },
        'tbbox': {
            "lx": 536982.3269807688,
            "ly": 2805172.6864022344,
            "lz": 228.40542941074819,
            "ux": 543434.6162458079,
            "uy": 2840735.7910883247,
            "uz": 1002.2817169420287
        }
    }

    # get all node
    # las_list = glob2.glob("%s/*.las"%src) #just first hierarchy
    las_list = glob2.glob("%s/**/*.las" % src)  # convert all

    if not las_list:
        print('can not find las')
        return

    tightBox = bbox.get('tbbox')
    mu = np.array([(tightBox['lx'] + tightBox['ux']) / 2, (tightBox['ly'] + tightBox['uy']) / 2,
                   (tightBox['lz'] + tightBox['uz']) / 2])

    # recode matrix
    mu = wgs84_from(*(mu), tm=proj_param)
    transM = wgs84_trans_matrix(*mu)

    def getkey(name):
        name = os.path.basename(name)
        return len(name)

    # sort as file name length
    las_list.sort(key=getkey)

    root = {
        'boundingVolume': {'box': []},
        'children': [],
        'content': {'url': ''},
        'geometricError': 0,
        'refine': 'ADD',
        'transform': list(transM.flatten()),  # r4x4, neu 2 wgs84
    }

    geomeotric_space = 16
    rootnode = treeNode()

    rootnode.setFile(las_list[0])
    for e in las_list:
        _node = treeNode()
        _node.setFile(e)
        rootnode.addNode(_node)

    pcd = read_las(rootnode.file, tm=proj_param)
    pcd = covert_neu(pcd, tm=proj_param, transM=transM)

    pcd2pnts(pcd, r'%s/%s.pnts' % (outdir, rootnode.key))
    root['boundingVolume']['box'] = pcd.get('neu').get('bbox')
    root['geometricError'] = pcd.get('neu').get('bbox')[3] / geomeotric_space
    root['content']['url'] = '%s.pnts' % (rootnode.key)

    visitNode(rootnode.childs, root['children'], proj_param, transM, outdir)
    tileset = {
        'asset': {'version': '0.0'},
        'geometricError': root['geometricError'],
        'root': root,
    }
    json.dump(tileset, open(r'%s/tileset.json' % outdir, 'w'))



#hierarchyStepSize=5

if __name__ == "__main__":
    testConvert()


