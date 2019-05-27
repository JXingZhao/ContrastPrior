import caffe
import numpy as np
class DepthContrastLoss(caffe.Layer):
    def setup(self, bottom, top):
        params = eval(self.param_str)
        # Due to that the gradient of this layer is too large, so we set lr_rate = 1e-13 to let the convergency more stable
        self.loss_weight = 1e-13
        self.variance_weight = params['variance_weight']
        self.distance_weight = params['distance_weight']
        if len(bottom) != 2:
            raise Exception("The number of bottom muse be two")
        if len(top) != 1:
            raise Exception("The number of top must be one")
    def reshape(self, bottom, top):
        top[0].reshape(1) 
    def forward(self, bottom, top):
        feature = bottom[0].data[0,0, ...]
        height, width = feature.shape
        label = bottom[1].data[0,0,...]
        NumSeg = len(np.unique(label))
        loss = 0.0
        self.full_std = feature.std()
        self.mean = []
        self.full_mean = feature.mean()
        self.std = []
        self.num = []
        if NumSeg != 2:
            return
        for s in range(NumSeg):
            area = np.where(label == s) 
            area_feature = feature[area]
            self.num.append(len(area_feature))
            self.mean.append(area_feature.mean())
            loss_t = area_feature.std()
            self.std.append(loss_t)
            loss += (-np.log(1 - loss_t) * self.variance_weight)
        
        mean_distance = (self.mean[0] - self.mean[1]) * (self.mean[0] - self.mean[1])
        
        if mean_distance == 0:
            mean_distance = mean_distance + 0.0001
        loss += (-np.log(mean_distance) * self.distance_weight)   
        top[0].data[...] = loss
        
                 
    def backward(self, top, propagate_down, bottom):
        diff = bottom[0].diff[0,0,...]
        

        feature = bottom[0].data[0,0, ...]
        height, width = diff.shape
        label = bottom[1].data[0,0,...]
        NumSeg = len(np.unique(label))
        if NumSeg != 2:
            return
        for s in range(NumSeg):
            area = np.where(label == s) 
            diff[area] -= ((1.0 / (1 - self.std[s]) * -((feature[area] - self.mean[s]) / self.num[s])) *  self.variance_weight * self.loss_weight)
        

        mean_distance = (self.mean[0] - self.mean[1]) * (self.mean[0] - self.mean[1])
        if mean_distance == 0:
            mean_distance = mean_distance + 0.0001
        diff[...] -= (-1) * (1 / (mean_distance)) * label / np.float(self.num[1]) * self.loss_weight * self.distance_weight
        diff[...] -= (1 / (mean_distance)) * (1 - label) / np.float(self.num[0]) * self.loss_weight * self.distance_weight
        
        
