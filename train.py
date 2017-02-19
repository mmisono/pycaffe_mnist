import numpy as np
import sys
sys.path.insert(0, "./caffe/python")
import caffe

def train(mode="gpu"):
    if mode == "gpu":
        caffe.set_mode_gpu()
        caffe.set_device(0)
    
    solver = caffe.get_solver('solver.prototxt')

    net = solver.net
    test_net = solver.test_nets[0]

    print(len(net._layer_names))
    for layer_name in net._layer_names:
        print(layer_name)
    print(len(net._blob_names))
    for blob_name in net._blob_names:
        print("{}: {}".format(blob_name,
            net.blobs[blob_name].data == test_net.blobs[blob_name].data))

    # this imitates solver.step()
    max_iter = 10000
    snapshot = 5000
    display = 100
    iter_size = 1
    test_interval = 500
    test_iter = 100
    iter_ = 0
    end = len(net.layers)-1
    while iter_ <= max_iter:
        net.clear_param_diffs()
        loss = 0
        acc = 0
        for i in range(iter_size):
            outs = net.forward()
            net.backward()
            loss += outs['softmax']
            acc += outs['accuracy']
        loss /= iter_size
        acc /= iter_size

        solver.apply_update()

        if iter_ > 0 and iter_ % display == 0:
            print("{}".format(iter_))
            print("    loss = {}".format(loss))
            print("    acc = {}".format(acc))
        
        if iter_ > 0 and  \
           test_interval > 0 and iter_ % test_interval == 0:
            acc = 0
            test_net.share_with(net)
            for i in range(test_iter):
                test_net.forward()
                acc += test_net.blobs['accuracy'].data
            acc /= test_iter
            print("    test acc = {}".format(acc))

        if iter_ > 0 and iter_ % snapshot == 0:
            solver.snapshot()

        iter_ += 1
        solver.set_iter(iter_)

    acc = 0
    test_net.share_with(net)
    for i in range(test_iter):
        test_net.forward()
        acc += test_net.blobs['accuracy'].data
    acc /= test_iter
    print("final test acc = {}".format(acc))

if __name__ == "__main__":
    train()
