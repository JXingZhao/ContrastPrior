caffe_root = '../../'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
from caffe import layers as L, params as P

weight_filler = dict(type='msra')
bias_filler = dict(type='constant', value=0)

def conv_bn_scale_relu(bottom, kernel_size, num_out, stride, pad, dilation=1):
    conv = L.Convolution(bottom, kernel_size=kernel_size, num_output=num_out, stride=stride,
                         pad=pad, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], 
                         weight_filler=weight_filler, bias_filler=bias_filler)
    bn = L.BatchNorm(conv, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)], 
                     use_global_stats=False, in_place=True)
    scale = L.Scale(conv, scale_param=dict(bias_term=True), in_place=True)
    relu = L.ReLU(conv, in_place=True)
    return conv, bn, scale, relu

def conv_bn_scale_relu_deploy(bottom, kernel_size, num_out, stride, pad, dilation=1):
    conv = L.Convolution(bottom, kernel_size=kernel_size, stride=stride, num_output=num_out,
                         pad=pad, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], 
                         weight_filler=weight_filler, bias_filler=bias_filler)
    bn = L.BatchNorm(conv, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)], 
                     use_global_stats=True, in_place=True)
    scale = L.Scale(conv, scale_param=dict(bias_term=True), in_place=True)
    relu = L.ReLU(conv, in_place=True)
    return conv, bn, scale, relu

def conv_bn_scale(bottom, kernel_size, num_out, stride, pad, dilation=1):
    conv = L.Convolution(bottom, kernel_size=kernel_size, num_output=num_out, stride=stride,
                         pad=pad, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], 
                         weight_filler=weight_filler, bias_filler=bias_filler)
    bn = L.BatchNorm(conv, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)], 
                     use_global_stats=False, in_place=True)
    scale = L.Scale(conv, scale_param=dict(bias_term=True), in_place=True)
    return conv, bn, scale

def conv_bn_scale_deploy(bottom, kernel_size, num_out, stride, pad, dilation=1):
    conv = L.Convolution(bottom, kernel_size=kernel_size, stride=stride, num_output=num_out,
                         pad=pad, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], 
                         weight_filler=weight_filler, bias_filler=bias_filler)
    bn = L.BatchNorm(conv, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)], 
                     use_global_stats=True, in_place=True)
    scale = L.Scale(conv, scale_param=dict(bias_term=True), in_place=True)
    return conv, bn, scale

def eltsum_relu(bottom1, bottom2):
    eltsum = L.Eltwise(bottom1, bottom2, eltwise_param=dict(operation=1))
    relu = L.ReLU(eltsum, in_place=True)
    return eltsum, relu

def identity_residual(bottom, kernel_size, num_out, stride, pad, dilation=1):
    conv1, bn1, scale1, relu1 = conv_bn_scale_relu(bottom, kernel_size, num_out, stride, pad)
    conv2, bn2, scale2 = conv_bn_scale(conv1, kernel_size, num_out, stride, pad)
    elt, elt_relu = eltsum_relu(bottom, conv2)
    return conv1, bn1, scale1, relu1, conv2, bn2, scale2, elt, elt_relu

def identity_residual_deploy(bottom, kernel_size, num_out, stride, pad, dilation=1):
    conv1, bn1, scale1, relu1 = conv_bn_scale_relu_deploy(bottom, kernel_size, num_out, stride, pad)
    conv2, bn2, scale2 = conv_bn_scale_deploy(conv1, kernel_size, num_out, stride, pad)
    elt, elt_relu = eltsum_relu(bottom, conv2)
    return conv1, bn1, scale1, relu1, conv2, bn2, scale2, elt, elt_relu

def project_residual(bottom, kernel_size, num_out, stride, pad, dilation=1):
    conv_proj, bn_proj, scale_proj = conv_bn_scale(bottom, 3, num_out, stride, 1)
    conv1, bn1, scale1, relu1 = conv_bn_scale_relu(bottom, kernel_size, num_out, stride, pad)
    conv2, bn2, scale2 = conv_bn_scale(conv1, kernel_size, num_out, 1, pad)
    elt, elt_relu = eltsum_relu(conv_proj, conv2)
    return conv_proj, bn_proj, scale_proj, conv1, bn1, scale1, relu1, conv2, bn2, scale2, elt, elt_relu

def project_residual_deploy(bottom, kernel_size, num_out, stride, pad, dilation=1):
    conv_proj, bn_proj, scale_proj = conv_bn_scale_deploy(bottom, 3, num_out, stride, 1)
    conv1, bn1, scale1, relu1 = conv_bn_scale_relu_deploy(bottom, kernel_size, num_out, stride, pad)
    conv2, bn2, scale2 = conv_bn_scale_deploy(conv1, kernel_size, num_out, 1, pad)
    elt, elt_relu = eltsum_relu(conv_proj, conv2)
    return conv_proj, bn_proj, scale_proj, conv1, bn1, scale1, relu1, conv2, bn2, scale2, elt, elt_relu

def make_resnet(training_data, mean_file, depth, num_channels):
    n = caffe.NetSpec()
    n.data, n.label = L.Data(source=training_data, backend=P.Data.LMDB, batch_size=128, ntop=2,
                                     transform_param=dict(crop_size=32, mean_file=mean_file, mirror=True),
                                     image_data_param=dict(shuffle=True), include=dict(phase=0))
    n.conv1, n.bn_conv1, n.scale_conv1, n.conv1_relu = conv_bn_scale_relu(n.data, 3, 16, 1, 1)
    last_stage = 'n.conv1'

    num_stages = len(num_channels)
    if (depth - 2) % (num_stages * 2) != 0:
        print '{} != {} * n + 2'.format(depth, num_stages)
        sys.exit()
    num_units = (depth - 2) / (num_stages * 2)
    for i in range(num_stages):
        num_channel = int(num_channels[i])
        for res in range(num_units):
            if res == 0:
                stride = 2
                if i == 0:
                    stride = 1
                res_uint = 'n.res%d%c_branch1,' % (i+2, res+97) + \
                           'n.bn%d%c_branch1,' % (i+2, res+97) + \
                           'n.scale%d%c_branch1,' % (i+2, res+97) + \
                           'n.res%d%c_branch2%c,' % (i+2, res+97, 97) + \
                           'n.bn%d%c_branch2%c,' % (i+2, res+97, 97) + \
                           'n.scale%d%c_branch2%c,' % (i+2, res+97, 97) + \
                           'n.res%d%c_branch2%c_relu,' % (i+2, res+97, 97) + \
                           'n.res%d%c_branch2%c,' % (i+2, res+97, 98) + \
                           'n.bn%d%c_branch2%c,' % (i+2, res+97, 98) + \
                           'n.scale%d%c_branch2%c,' % (i+2, res+97, 98) + \
                           'n.res%d%c,' % (i+2, res+97) + \
                           'n.res%d%c_relu' % (i+2, res+97) + \
                           ' = project_residual(' + last_stage + ', 3, num_channel, stride, 1)'
                exec(res_uint)
                last_stage = 'n.res%d%c' % (i+2, res+97)
                continue

            res_uint = 'n.res%d%c_branch2%c,' % (i+2, res+97, 97) + \
                       'n.bn%d%c_branch2%c,' % (i+2, res+97, 97) + \
                       'n.scale%d%c_branch2%c,' % (i+2, res+97, 97) + \
                       'n.res%d%c_branch2%c_relu,' % (i+2, res+97, 97) + \
                       'n.res%d%c_branch2%c,' % (i+2, res+97, 98) + \
                       'n.bn%d%c_branch2%c,' % (i+2, res+97, 98) + \
                       'n.scale%d%c_branch2%c,' % (i+2, res+97, 98) + \
                       'n.res%d%c,' % (i+2, res+97) + \
                       'n.res%d%c_relu' % (i+2, res+97) + \
                       ' = identity_residual(' + last_stage + ', 3, num_channel, 1, 1)'
            exec(res_uint)
            last_stage = 'n.res%d%c' % (i+2, res+97)
            
    exec('n.pool_global = L.Pooling(' + last_stage + ', pool=P.Pooling.AVE, global_pooling=True)')
    n.score = L.InnerProduct(n.pool_global, num_output=10, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
    n.prob = L.SoftmaxWithLoss(n.score, n.label)
    n.acc = L.Accuracy(n.score, n.label)

    return n.to_proto()
def make_resnet_deploy(test_data, mean_file, depth, num_channels):
    n = caffe.NetSpec()
    n.data, n.label = L.Data(source=test_data, backend=P.Data.LMDB, batch_size=100, ntop=2,
            transform_param=dict(crop_size=32, mean_file=mean_file, mirror=False),
            include=dict(phase=1))
    n.conv1, n.bn_conv1, n.scale_conv1, n.conv1_relu = conv_bn_scale_relu_deploy(n.data, 3, 16, 1, pad=1)
    last_stage = 'n.conv1'

    num_stages = len(num_channels)
    if (depth - 2) % (num_stages * 2) != 0:
        print '{} != {} * n + 2'.format(depth, num_stages)
        sys.exit()
    num_units = (depth - 2) / (num_stages * 2)
    for i in range(num_stages):
        num_channel = int(num_channels[i])
        for res in range(num_units):
            if res == 0:
                stride = 2
                if i == 0:
                    stride = 1
                res_uint = 'n.res%d%c_branch1,' % (i+2, res+97) + \
                           'n.bn%d%c_branch1,' % (i+2, res+97) + \
                           'n.scale%d%c_branch1,' % (i+2, res+97) + \
                           'n.res%d%c_branch2%c,' % (i+2, res+97, 97) + \
                           'n.bn%d%c_branch2%c,' % (i+2, res+97, 97) + \
                           'n.scale%d%c_branch2%c,' % (i+2, res+97, 97) + \
                           'n.res%d%c_branch2%c_relu,' % (i+2, res+97, 97) + \
                           'n.res%d%c_branch2%c,' % (i+2, res+97, 98) + \
                           'n.bn%d%c_branch2%c,' % (i+2, res+97, 98) + \
                           'n.scale%d%c_branch2%c,' % (i+2, res+97, 98) + \
                           'n.res%d%c,' % (i+2, res+97) + \
                           'n.res%d%c_relu' % (i+2, res+97) + \
                           ' = project_residual_deploy(' + last_stage + ', 3, num_channel, stride, 1)'
                exec(res_uint)
                last_stage = 'n.res%d%c' % (i+2, res+97)
                continue

            res_uint = 'n.res%d%c_branch2%c,' % (i+2, res+97, 97) + \
                       'n.bn%d%c_branch2%c,' % (i+2, res+97, 97) + \
                       'n.scale%d%c_branch2%c,' % (i+2, res+97, 97) + \
                       'n.res%d%c_branch2%c_relu,' % (i+2, res+97, 97) + \
                       'n.res%d%c_branch2%c,' % (i+2, res+97, 98) + \
                       'n.bn%d%c_branch2%c,' % (i+2, res+97, 98) + \
                       'n.scale%d%c_branch2%c,' % (i+2, res+97, 98) + \
                       'n.res%d%c,' % (i+2, res+97) + \
                       'n.res%d%c_relu' % (i+2, res+97) + \
                       ' = identity_residual_deploy(' + last_stage + ', 3, num_channel, 1, 1)'
            exec(res_uint)
            last_stage = 'n.res%d%c' % (i+2, res+97)
            
    exec('n.pool_global = L.Pooling(' + last_stage + ', pool=P.Pooling.AVE, global_pooling=True)')
    n.score = L.InnerProduct(n.pool_global, num_output=10,
                                          param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                          weight_filler=dict(type='gaussian', std=0.01),
                                          bias_filler=dict(type='constant', value=0))
    n.acc = L.Accuracy(n.score, n.label)

    return n.to_proto()

if __name__ == '__main__':
    train_data_path = '/home/andrew/Projects/caffe/examples/cifar10/cifar10_train_lmdb'
    test_data_path = '/home/andrew/Projects/caffe/examples/cifar10/cifar10_test_lmdb'
    mean_file_path = '/home/andrew/Projects/caffe/examples/cifar10/mean.binaryproto'
    depth =  int(sys.argv[1])
    num_channels = [16, 32, 64]
    train_prototxt = str(make_resnet(train_data_path, mean_file_path, depth, num_channels))
    deploy_prototxt = str(make_resnet_deploy(test_data_path, mean_file_path, depth, num_channels))

    train_file = './resnet_cifar10_{}.prototxt'.format(depth)
    deploy_file = './resnet_cifar10_{}_deploy.prototxt'.format(depth)
    with open(train_file, 'w') as f:
        f.write(train_prototxt)
    with open(deploy_file, 'w') as f:
        f.write(deploy_prototxt)
    print('Saved ' + train_file)
    print('Saved ' + deploy_file)
