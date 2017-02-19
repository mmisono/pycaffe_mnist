# MNIST training examples with pycaffe

## Prepare
```sh
$ cd caffe/
$ patch -p1 < ../pycaffe.diff
```

## Generate prototxt
```
python net.py
```

`train.prototxt`, `test.prototxt`, `deploy.prottxt` will be generated.

## Prepare dataset
```sh
$ cd ./caffe/
$ ./data/mnist/get_mnist.sh
$ ./examples/mnist/create_mnist.sh
$ mv ./caffe/examples/mnist/mnist_{test,train}_lmdb .
```

## Training
```sh
$ python train.py
```
