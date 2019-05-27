import caffe
import numpy as np
from PIL import Image
import random
import cv2

class ImageLabelData(caffe.Layer):
    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.root_folder = params['root_folder']
        self.source = params['source']
        self.seed = params['seed']
        self.mirror = params.get('mirror', False)
        self.random = params.get('random', False)
        self.mir = 0
        self.mean = params['mean']
        self.new_width = params.get('new_width', 0)
        self.new_height = params.get('new_height', 0)
        self.intlen = 400
        self.floatlen = 400.0

        
        
        if len(bottom) != 0:
            raise Exception("Dont need to define bottom")
        self.indices = open(self.source).read().splitlines()
        self.imglist = []
        self.gtlist = []
        self.depthlist = []
        self.num_images = len(self.indices)
        
        
        
 
        for line in self.indices:
            impath = line.split()[0] 
            self.imglist.append('/LR/' + impath[8:])
            if len(line.split()) > 1:
                gtpath = line.split()[1]
                self.gtlist.append(gtpath)
                    #self.depthlist.append(impath)
                self.depthlist.append('/depth/' + impath[8:])

                
                
        if len(self.imglist) != len(self.indices):
            raise Exception("imglist wrong")
        print('total image:', len(self.imglist))
        self.idx = 0
        self.perm = range(self.num_images)
        if self.random:
            random.seed(self.seed)
            random.shuffle(self.perm)
        
        
    def reshape(self, bottom, top):
        self.data = self.load_image(self.idx)
        top[0].reshape(1, *self.data.shape)
        if len(top) > 1:
            self.gt = self.load_gt(self.idx)
            top[1].reshape(1, 1, *self.gt.shape)
        if len(top) > 2:
            self.depth = self.load_depth(self.idx)
            top[2].reshape(1,  *self.depth.shape)
        if len(top) > 3:
            self.load_gt16(self.idx)
            
            top[3].reshape(1, 1, *self.gt2.shape)
        if len(top) > 4:
            
            top[4].reshape(1, 1, *self.gt4.shape)    
        if len(top) > 5:
       
            top[5].reshape(1, 1, *self.gt8.shape)  
        if len(top) > 6:
       
            top[6].reshape(1, 1, *self.gt16.shape) 
        if len(top) > 7:
       
            top[7].reshape(1, 1, *self.gt32.shape) 
        

    def load_image(self, idx):
        im = cv2.imread(self.root_folder + self.imglist[self.perm[idx]])
        #print self.root_folder + self.imglist[self.perm[idx]]
        #print self.root_folder + self.imglist[self.perm[idx]]
        #print self.root_folder + self.imglist[self.perm[idx]]
        if self.new_width != 0 :
            im = cv2.resize(im, (self.new_height, self.new_width))
        if im.shape[0] > self.intlen:
            im = cv2.resize(im, (np.int(im.shape[1] * (self.floatlen / im.shape[0])), self.intlen))
        if im.shape[1] > self.intlen:
            im = cv2.resize(im, (self.intlen, np.int(im.shape[0] * (self.floatlen / im.shape[1]))))
        #print self.imglist[idx]
        in_ = np.array(im, dtype=np.float32)
        in_ -= self.mean
        in_ = in_[:,:,::-1] 
        #print in_.shape
        if self.mir == 1:
            in_ = in_[:, ::-1,:]
        
        in_ = in_.transpose((2,0,1))
        return in_
    
    def load_gt(self, idx):
        im = cv2.imread(self.root_folder + self.gtlist[self.perm[idx]], 0)
        
        im = im / 255
        #print np.max(im), np.min(im)
        #print self.root_folder + self.imglist[self.perm[idx]]
        #print self.root_folder + self.imglist[self.perm[idx]]
        if self.new_width != 0 :
            im = cv2.resize(im, (self.new_height, self.new_width))
        if im.shape[0] > self.intlen:
            im = cv2.resize(im, (np.int(im.shape[1] * (self.floatlen / im.shape[0])), self.intlen))
        if im.shape[1] > self.intlen:
            im = cv2.resize(im, (self.intlen, np.int(im.shape[0] * (self.floatlen / im.shape[1]))))
        #print self.imglist[idx]
        in_ = np.array(im, dtype=np.float32)
        if self.mir == 1:
            in_ = in_[:, ::-1]
        if in_.ndim != 2:
            raise Exception("gt image has one more dimention")
        return in_
    
    

    def load_gt16(self, idx):
        im = cv2.imread(self.root_folder + self.gtlist[self.perm[idx]], 0)
        
        im = im / 255
        #print self.root_folder + self.imglist[self.perm[idx]]
        #print self.root_folder + self.imglist[self.perm[idx]]
        if self.new_width != 0 :
            im = cv2.resize(im, (self.new_height, self.new_width))
        if im.shape[0] > 400:
            im = cv2.resize(im, (np.int(im.shape[1] * (400.0 / im.shape[0])), 400))
        if im.shape[1] > 400:
            im = cv2.resize(im, (400, np.int(im.shape[0] * (400.0 / im.shape[1]))))
        #print self.imglist[idx]
        self.gt2 = cv2.resize(im, (np.int(np.floor(im.shape[1] / 2.0)), np.int(np.floor(im.shape[0] / 2.0))))
        self.gt4 = cv2.resize(im, (np.int(np.floor(self.gt2.shape[1] / 2.0)), np.int(np.floor(self.gt2.shape[0] / 2.0))))
        self.gt8 = cv2.resize(im, (np.int(np.floor(self.gt4.shape[1] / 2.0)), np.int(np.floor(self.gt4.shape[0] / 2.0))))
        self.gt16 = cv2.resize(im, (np.int(np.floor(self.gt8.shape[1] / 2.0)), np.int(np.floor(self.gt8.shape[0] / 2.0))))
        self.gt32 = cv2.resize(im, (np.int(np.floor(self.gt16.shape[1] / 2.0)), np.int(np.floor(self.gt16.shape[0] / 2.0))))
        self.gt2 = np.array(self.gt2, dtype=np.float32)
        self.gt4 = np.array(self.gt4, dtype=np.float32)
        self.gt8 = np.array(self.gt8, dtype=np.float32)
        self.gt16 = np.array(self.gt16, dtype=np.float32)
        self.gt32 = np.array(self.gt32, dtype=np.float32)
        if self.mir == 1:
            self.gt2 = self.gt2[:, ::-1]
            self.gt4 = self.gt4[:, ::-1]
            self.gt8 = self.gt8[:, ::-1]
            self.gt16 = self.gt16[:, ::-1]
            self.gt32 = self.gt32[:, ::-1]
        
        
    def load_depth(self, idx):
        im = cv2.imread(self.root_folder + self.depthlist[self.perm[idx]])
        
        #print self.root_folder + self.depthlist[self.perm[idx]]
        #print self.root_folder + self.imglist[self.perm[idx]]
       # print self.root_folder + self.depthlist[self.perm[idx]]
        if self.new_width != 0 :
            im = cv2.resize(im, (self.new_height, self.new_width))
        if im.shape[0] > self.intlen:
            im = cv2.resize(im, (np.int(im.shape[1] * (self.floatlen / im.shape[0])), self.intlen))
        if im.shape[1] > self.intlen:
            im = cv2.resize(im, (self.intlen, np.int(im.shape[0] * (self.floatlen / im.shape[1]))))
        #im = cv2.resize(im, (np.int(im.shape[1] / 2.0), np.int(im.shape[0] / 2.0)))
        #print self.imglist[idx]
        in_ = np.array(im, dtype=np.float32)
        #in_ = 255 - in_ 
        #in_ -= 119.96
        #in_ = in_ / 255.0
        #print in_.shape
        
        in_ -= self.mean
        in_ = in_[:,:,::-1] 
      #  print in_.shape
        if self.mir == 1:
            in_ = in_[:, ::-1]
        
        in_ = in_.transpose((2,0,1))
        
        
        return in_
        
       
    def forward(self, bottom, top):
        top[0].data[...] = self.data
        if len(top) > 1:
            top[1].data[...] = self.gt
        if len(top) > 2:
            top[2].data[...] = self.depth
        if len(top) > 3:
            top[3].data[...] = self.gt2
        if len(top) > 4:
            top[4].data[...] = self.gt4
        if len(top) > 5:
            top[5].data[...] = self.gt8  
        if len(top) > 6:
            top[6].data[...] = self.gt16  
        if len(top) > 7:
            top[7].data[...] = self.gt32  
        #print self.id
        if self.mirror == True:
            self.mir = random.randint(0, 1)
        self.idx += 1
        if self.idx == self.num_images:
            random.shuffle(self.perm)
            self.idx = 0
            #print self.perm
        #im_name = 'test/%6d.png' % (self.idx)
        #cv2.imwrite(im_name, self.label * 40)
        

        
    def backward(self, top, propagate_down, bottom):
        pass












