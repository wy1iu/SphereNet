#!/bin/bash

if [ ! -d "cifar10" ]; then
  mkdir cifar10
fi
if [ ! -d "models" ]; then
  mkdir models
fi
cd cifar10
wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar zxvf cifar-10-binary.tar.gz
cp -r cifar-10-batches-bin/* ./
rm -r cifar-10-batches-bin/
rm cifar-10-binary.tar.gz


