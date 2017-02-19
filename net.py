import os.path as osp
import sys
sys.path.insert(0, "./caffe/python")

import caffe
from caffe import layers as L, params as P
from caffe_layer_wrapper import *

# lenet
def create_proto(split):
    n = caffe.NetSpec()

    # data layer
    if split == 'train' or split == 'test':
        if split == 'train':
            source = "mnist_train_lmdb"
            phase = caffe.TRAIN
            bs = 64
        else:
            source = "mnist_test_lmdb"
            phase = caffe.TEST
            bs = 100

        data_param = {
                "source" : source,
                "backend" :  P.Data.LMDB,
                "batch_size" : bs,
        }
        transform_param = {
                "scale": 1/256.0
        }
        n.data, n.label = L.Data(
            ntop=2,  data_param=data_param, transform_param = transform_param)
    elif split == 'deploy':
        n.data = L.Input(
            input_param={"shape": {"dim": [64, 1, 28, 28]}})
    else:
        raise NotImplementedError

    n.conv1 = convolution(n.data, nout=20, ks=5, stride=1, pad=0)
    n.pool1 = max_pool(n.conv1, ks=2, stride=2)
    n.conv2 = convolution(n.pool1, nout=50, ks=5, stride=1, pad=0)
    n.pool2 = max_pool(n.conv2, ks=2, stride=2)
    n.ip1   = inner_product(n.pool2, nout=500)
    n.relu1 = relu(n.ip1)
    n.ip2   = inner_product(n.ip1, nout=10)
               
    if split == 'train' or split == 'test':
        n.accuracy = accuracy(n.ip2, n.label)
        n.softmax = softmax_loss(n.ip2, n.label)
    elif split == 'deploy':
        n.softmax = softmax(n.ip2)

    return n.to_proto()

def save_proto(split):
    with open('{}.prototxt'.format(split), 'w') as f:
        f.write(str(create_proto(split)))
    print("create {}.prototxt".format(split))


if __name__ == '__main__':
    save_proto("train")
    save_proto("test")
    save_proto("deploy")
