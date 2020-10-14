from abc import ABC

import tensorflow as tf
from tensorflow.keras.models import Model
import tensorflow.keras.layers as layers


class Conv2D_BN(layers.Layer):
    def __init__(self, filters, kernel_size, name, strides=(1, 1), padding="valid", use_bias=True):
        super(Conv2D_BN, self).__init__()
        self.conv2d = layers.Conv2D(filters=filters,
                                    kernel_size=kernel_size,
                                    strides=strides,
                                    padding=padding,
                                    use_bias=use_bias,
                                    data_format="channels_last",
                                    name=name + "_convlayer")
        self.bn = layers.BatchNormalization(momentum=0.0, name=name + "_bnlayer")
        self.activation = layers.Activation("relu", name=name + "_actilayer")

    def call(self, inputs, training=None):
        x = self.conv2d(inputs)
        x = self.bn(x, training=training)
        x = self.activation(x)
        return x


class Dense_BN(layers.Layer):
    def __init__(self, units, name, use_bias=True):
        super(Dense_BN, self).__init__()
        self.dense = layers.Dense(units=units,
                                  use_bias=use_bias,
                                  name=name + "_denselayer")
        self.bn = layers.BatchNormalization(momentum=0.0, name=name + "_bnlayer")
        self.activation = layers.Activation("relu", name=name + "_actilayer")

    def call(self, inputs, training=None):
        x = self.dense(inputs)
        x = self.bn(x, training=training)
        x = self.activation(x)
        return x


class PointNet(Model, ABC):

    def __init__(self, num_point, num_attribute, num_classes):
        super(PointNet, self).__init__()
        self.num_point = num_point
        self.num_attribute = num_attribute
        self.num_classes = num_classes
        self.conv1 = Conv2D_BN(64, [1, self.num_attribute], "conv1")
        self.conv2 = Conv2D_BN(64, [1, 1], "conv2")
        self.conv3 = Conv2D_BN(64, [1, 1], "conv3")
        self.conv4 = Conv2D_BN(128, [1, 1], "conv4")
        self.conv5 = Conv2D_BN(1024, [1, 1], "conv5")
        self.maxpool1 = layers.MaxPool2D(pool_size=[self.num_point, 1], name="maxpool1")
        self.flaten1 = layers.Flatten(data_format="channels_last")
        self.fc1 = Dense_BN(256, "fc1")
        self.fc2 = Dense_BN(128, "fc2")
        self.conv6 = Conv2D_BN(512, [1, 1], "conv6")
        self.conv7 = Conv2D_BN(256, [1, 1], "conv7")
        self.dp1 = layers.Dropout(0.3, name="dp1")
        self.output_layer = layers.Conv2D(self.num_classes, [1, 1], [1, 1], "valid", use_bias=False,
                                          name="output_layer")

    def call(self, inputs, training=None, mask=None):
        # print("inputs_shape",inputs.shape)
        batch_size = inputs.shape[0]  # i take whole dat to debug this
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)
        point_feat1 = self.conv5(x, training=training)
        # print("feat1_shape",point_feat1.shape)
        x = self.maxpool1(point_feat1)
        x = self.flaten1(x)
        x = self.fc1(x, training=training)
        x = self.fc2(x, training=training)
        x = tf.reshape(x, [batch_size, 1, 1, -1])
        x = tf.tile(x, [1, self.num_point, 1, 1])
        # print("tile_shape",x.shape)
        x = tf.concat(axis=3, values=[point_feat1, x])
        x = self.conv6(x, training=training)
        x = self.conv7(x, training=training)
        if training:
            x = self.dp1(x, training=training)
        x = self.output_layer(x)
        x = tf.squeeze(x, axis=2)
        return x


class PointNetLoss(tf.keras.losses.Loss, ABC):

    def __init__(self, reduction='NONE'):
        if reduction is 'NONE':
            super(PointNetLoss, self).__init__(reduction=tf.keras.losses.Reduction.NONE)
        else:
            super(PointNetLoss, self).__init__()

    def call(self, y_true, y_pred):
        return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
