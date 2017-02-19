import caffe
from caffe import layers as L, params as P

def inner_product(bottom, nout,
                  weight_filler=dict(type="xavier"),
                  bias_filler=dict(type="constant"),
                  param=[dict(lr_mult=1, decay_mult=1),
                         dict(lr_mult=2, decay_mult=0)]):
    return L.InnerProduct(bottom, num_output=nout,
                          weight_filler = weight_filler,
                          bias_filler = bias_filler,
                          param = param)
def relu(bottom):
    return L.ReLU(bottom, in_place=True)

def convolution(bottom, nout, ks=3, stride=1, pad=1,
                weight_filler=dict(type="xavier"),
                bias_filler=dict(type="constant"),
                param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         weight_filler = weight_filler,
                         bias_filler = bias_filler,
                         num_output=nout, pad=pad, param=param)
    return conv

def conv_relu(bottom, nout, ks=3, stride=1, pad=1,
              param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]):
    conv = convolution(bottom, nout, ks, stride, pad, param)
    return conv, L.ReLU(conv, in_place=True)

def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def softmax(bottom):
    return L.Softmax(bottom)

def softmax_loss(bottom, label, loss_weight=1):
    return L.SoftmaxWithLoss(bottom, label, loss_weight=1)

def accuracy(bottom, label, test_only = False):
    if test_only:
        return L.Accuracy(bottom, label, include=dict(phase=caffe.TEST))
    return L.Accuracy(bottom, label)
