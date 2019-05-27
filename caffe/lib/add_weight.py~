import caffe
import numpy as np
import cv2

class addWeight(caffe.Layer):
    """
    bottom[0]: feature map
    bottom[1]: weight map
    """
    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("The number of bottom muse be two")

        if len(top) != 1:
            raise Exception("The number of top must be one")

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].shape)

    def forward(self, bottom, top):
        
        self.bottom_1 = bottom[1].data
        self.bottom_0 = bottom[0].data


        top[0].data[...] = bottom[1].data * bottom[0].data

    def backward(self, top, propagate_down, bottom):
        tmp = np.ones(bottom[0].data.shape)
        bottom[0].diff[...] = tmp * self.bottom_1 * top[0].diff
        bottom[1].diff[...] = (top[0].diff * self.bottom_0).sum(axis=1)[np.newaxis, ...]
