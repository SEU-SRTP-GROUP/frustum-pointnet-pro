''' Frsutum PointNets v1 Model.
'''
from __future__ import print_function

import sys
import os
import tensorflow as tf
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import tf_util
from model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, NUM_OBJECT_POINT
from model_util import point_cloud_masking, get_center_regression_net
from model_util import placeholder_inputs, parse_output_to_tensors, get_loss
from model_util import placeholder_inputs, parse_output_to_tensors, get_loss
from model_util import extract_h2_features
def get_instance_seg_v1_net(point_cloud, one_hot_vec,
                            is_training, bn_decay, end_points):
    ''' 3D instance segmentation PointNet v1 network.
    Input:
        point_cloud: TF tensor in shape (B,N,4)
            frustum point clouds with XYZ and intensity in point channels
            XYZs are in frustum coordinate
        one_hot_vec: TF tensor in shape (B,3)
            length-3 vectors indicating predicted object type
        is_training: TF boolean scalar
        bn_decay: TF float scalar
        end_points: dict
    Output:
        logits: TF tensor in shape (B,N,2), scores for bkg/clutter and object
        end_points: dict
    '''
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value

    net = tf.expand_dims(point_cloud, 2)

    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)
    point_feat = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(point_feat, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)
    global_feat = tf_util.max_pool2d(net, [num_point,1],
                                     padding='VALID', scope='maxpool')

    global_feat = tf.concat([global_feat, tf.expand_dims(tf.expand_dims(one_hot_vec, 1), 1)], axis=3)
    global_feat_expand = tf.tile(global_feat, [1, num_point, 1, 1])
    concat_feat = tf.concat(axis=3, values=[point_feat, global_feat_expand])

    net = tf_util.conv2d(concat_feat, 512, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv6', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 256, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv7', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv8', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv9', bn_decay=bn_decay)
    net = tf_util.dropout(net, is_training, 'dp1', keep_prob=0.5)

    logits = tf_util.conv2d(net, 2, [1,1],
                         padding='VALID', stride=[1,1], activation_fn=None,
                         scope='conv10')
    logits = tf.squeeze(logits, [2]) # BxNxC
    return logits, end_points
 

def get_3d_box_estimation_v1_net(object_point_cloud, one_hot_vec,
                                 is_training, bn_decay, end_points):
    ''' 3D Box Estimation PointNet v1 network.
    Input:
        object_point_cloud: TF tensor in shape (B,M,C)
            point clouds in object coordinate
        one_hot_vec: TF tensor in shape (B,3)
            length-3 vectors indicating predicted object type
    Output:
        output: TF tensor in shape (B,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4)
            including box centers, heading bin class scores and residuals,
            and size cluster scores and residuals
    '''
    '''
    这里对原本的网络进行修改，为了更好地是西安学习shape,我们将对每一个point求其对与中心点的 二范数，然后用一个MLP提取其特征，特征为（1，3）最后将所有的特征整合成为一张特征图
    再送到网络里去
    '''
    num_point = object_point_cloud.get_shape()[1].value

    batch_size = object_point_cloud.get_shape()[0].value
    channels =  object_point_cloud.get_shape()[2].value
    #识别 dimension, center
    net0=object_point_cloud
    result_feature = tf.TensorArray(size=0 , dtype=tf.float32, dynamic_size=True)


    # def cond_nump(batch,num,num_points,n, channels,result_feature,net0):
    #     return num<num_points
    #
    # def body_nump(batch,num,num_points,n, channels,result_feature,net0):
    #     print("#################################",num)
    #     feature = tf.slice(net0, [0+batch,0+num,0], [1,1,channels])
    #     feature = tf.norm(tf.squeeze(feature, axis=0))
    #     feature = extract_h2_features(tf.expand_dims(feature, axis=0), 'extract_h2', 'extractor',[4,16,32,16,channels])
    #     result_feature=result_feature.write(n, feature)
    #     return batch, num + 1, num_points,n + 1,channels, result_feature, net0
    #
    # def cond_batch(batch,batch_size,num_points,n,channels,result_feature,net0):
    #     return batch<batch_size
    #
    # def body_batch(batch, batch_size, num_points, n, channels, result_feature, net0):
    #     print("#################################", batch)
    #     batch,num,num_points,n, channels,result_feature,net0=tf.while_loop(cond_nump, body_nump, [batch,0,num_points,n,channels,result_feature,net0])
    #     return batch+1,batch_size,num_points,n,channels,result_feature,net0
    #
    # _, _, _, _, _, result_feature, _= tf.while_loop(cond_batch,body_batch,[0, batch_size, num_point, 0, channels, result_feature, net0])
    def cond_num(num,num_point,channels,result_feature,net0):
        return num<num_point
    def body_num(num,num_point,channels,result_feature,net0):
        feature_abc=[]
        if num_point-num < 3 :
            feature_abc=tf.slice(net0, [0, 0+num, 0], [-1, 1, channels])
            feature_abc = tf.norm(feature_abc, axis=2)
            feature = extract_h2_features(feature_abc, 'extract_h1', 'extractor', [4, 16, 32, 16, 3])
        else:
            feature_a = tf.slice(net0, [0, 0+num, 0], [-1, 1, channels])  #[batch,1,3]
            feature_a = tf.norm(feature_a, axis=2)   #[batch,1]
            feature_b = tf.slice(net0, [0, 0+num+1, 0], [-1, 1, channels])  # [batch,1,3]
            feature_b = tf.norm(feature_b, axis=2)  # [batch,1]
            feature_c = tf.slice(net0, [0, 0+num+2, 0], [-1, 1, channels])  # [batch,1,3]
            feature_c = tf.norm(feature_c, axis=2)  # [batch,1]
            h_ab = tf.norm(feature_a - feature_b, axis=2)
            h_bc = tf.norm(feature_b - feature_c, axis=2)
            h_ca = tf.norm(feature_c - feature_a, axis=2)
            feature_abc = feature_abc.write(0, feature_a)
            feature_abc = feature_abc.write(1, h_ab)
            feature_abc = feature_abc.write(2, feature_b)
            feature_abc = feature_abc.write(3, h_bc)
            feature_abc = feature_abc.write(4, feature_c)
            feature_abc = feature_abc.write(5, h_ca)
            feature = extract_h2_features(feature_abc,'extract_h2','extractor',[16,32,64,32,9])   # [batch,channels]
        result_feature = result_feature.write(num, feature)

        return num+3,num_point,channels,result_feature,net0

    _,_,_,result_feature,_=tf.while_loop(cond_num,body_num,[0,num_point,channels,result_feature,net0])
    net0 = tf.transpose(result_feature.stack(),[1,0,2])   # result_feature.stack() (num_point,batch,channels) 所以需要转置
    net0=   tf.reshape( net0,[batch_size,num_point,channels])
    print(net0.get_shape().as_list(),"############################################################")

    # result_feature=[]
    # for i in range(batch_size):
    #     print("batch="+str(i)+"#####################################################")
    #     for j in range(num_point):
    #         print("num_point="+str(j)+"#####################################################")
    #         feature =  tf.slice(net0, [0+i,0+j,0], [1,1,channels]) # (N,1)
    #         feature =tf.norm(tf.squeeze(feature,axis=0))
    #         feature = extract_h2_features(tf.expand_dims(feature,axis=0),'extract_h2','extractor')
    #         result_feature.append(feature)
    print('阶段1###########################################################################')
    # net0= tf.reshape(tf.concat(result_feature,axis=0),[batch_size,num_point,-1])
    print('阶段2###########################################################################')
    net0 = tf.expand_dims( net0, 2)
    print(net0.get_shape().as_list(),'######################################################')
    net0 = tf_util.conv2d(net0, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv-reg1_s', bn_decay=bn_decay)
    net0 = tf_util.conv2d(net0, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv-reg2_s', bn_decay=bn_decay)
    net0= tf_util.conv2d(net0, 256, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv-reg3_s', bn_decay=bn_decay)
    net0 = tf_util.conv2d(net0, 512, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv-reg4_s', bn_decay=bn_decay)
    net0 = tf_util.max_pool2d(net0, [num_point, 1],
                             padding='VALID', scope='maxpool2_s')
    net0 = tf.squeeze(net0, axis=[1, 2])
    net0 = tf.concat([net0, one_hot_vec], axis=1)
    net0 = tf_util.fully_connected(net0, 512, scope='fc1_s', bn=True,
                                  is_training=is_training, bn_decay=bn_decay)
    net0 = tf_util.fully_connected(net0, 256, scope='fc2_s', bn=True,
                                  is_training=is_training, bn_decay=bn_decay)
    output_size = tf_util.fully_connected(net0,
                                             NUM_SIZE_CLUSTER * 4, activation_fn=None,
                                             scope='fc3_s')

    # output_center = tf.slice(  output_size_center , [0,0], [-1,3])
    # output_size =  tf.slice(  output_size_center , [0,3], [-1,NUM_SIZE_CLUSTER * 4])
    print('阶段3###########################################################################')
    net = tf.expand_dims(object_point_cloud, 2)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv-reg1_h', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv-reg2_h', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 256, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv-reg3_h', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 512, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv-reg4_h', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point,1],
        padding='VALID', scope='maxpool2_h')
    net = tf.squeeze(net, axis=[1,2])
    net = tf.concat([net, one_hot_vec], axis=1)
    net = tf_util.fully_connected(net, 512, scope='fc1_h', bn=True,
        is_training=is_training, bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, scope='fc2_h', bn=True,
        is_training=is_training, bn_decay=bn_decay)

    # The first 3 numbers: box center coordinates (cx,cy,cz),
    # the next NUM_HEADING_BIN*2:  heading bin class scores and bin residuals
    # next NUM_SIZE_CLUSTER*4: box cluster scores and residuals
    output_center_heading = tf_util.fully_connected(net,
        3+NUM_HEADING_BIN*2, activation_fn=None, scope='fc3_h')
    output = tf.concat([output_center_heading,output_size],axis=-1)
    print('阶段4###########################################################################')
    return  output,end_points


def get_model(point_cloud, one_hot_vec, is_training, bn_decay=None):
    ''' Frustum PointNets model. The model predict 3D object masks and
    amodel bounding boxes for objects in frustum point clouds.

    Input:
        point_cloud: TF tensor in shape (B,N,4)
            frustum point clouds with XYZ and intensity in point channels
            XYZs are in frustum coordinate
        one_hot_vec: TF tensor in shape (B,3)
            length-3 vectors indicating predicted object type
        is_training: TF boolean scalar
        bn_decay: TF float scalar
    Output:
        end_points: dict (map from name strings to TF tensors)
    '''
    end_points = {}
    
    # 3D Instance Segmentation PointNet
    logits, end_points = get_instance_seg_v1_net(\
        point_cloud, one_hot_vec,
        is_training, bn_decay, end_points)
    end_points['mask_logits'] = logits

    # Masking
    # select masked points and translate to masked points' centroid
    object_point_cloud_xyz, mask_xyz_mean, end_points = \
        point_cloud_masking(point_cloud, logits, end_points)

    # T-Net and coordinate translation
    center_delta, end_points = get_center_regression_net(\
        object_point_cloud_xyz, one_hot_vec,
        is_training, bn_decay, end_points)
    stage1_center = center_delta + mask_xyz_mean # Bx3
    end_points['stage1_center'] = stage1_center
    # Get object point cloud in object coordinate
    object_point_cloud_xyz_new = \
        object_point_cloud_xyz - tf.expand_dims(center_delta, 1)

    # Amodel Box Estimation PointNet
    output, end_points = get_3d_box_estimation_v1_net(\
        object_point_cloud_xyz_new, one_hot_vec,
        is_training, bn_decay, end_points)

    # Parse output to 3D box parameters
    end_points = parse_output_to_tensors(output, end_points)
    end_points['center'] = end_points['center_boxnet'] + stage1_center # Bx3

    return end_points


if __name__=='__main__':
    with tf.Graph().as_default():

        inputs = tf.zeros((32,1024,4))
        outputs = get_model(inputs, tf.ones((32,3)), tf.constant(True))
        variable_names = [v.name for v in tf.trainable_variables()]
        print(variable_names)
        for key in outputs:
            print((key, outputs[key]))
        loss = get_loss(tf.zeros((32,1024),dtype=tf.int32),
            tf.zeros((32,3)), tf.zeros((32,),dtype=tf.int32),
            tf.zeros((32,)), tf.zeros((32,),dtype=tf.int32),
            tf.zeros((32,3)), outputs)
        print(loss)
