import hyper_parameter
import provider
import model

import numpy as np
import tensorflow as tf
import xlwt


def load_data(test_area=6):
    # 读取点云数据
    data_batch_list = []
    label_batch_list = []
    all_files = provider.getDataFiles(hyper_parameter.ALL_FILES_PATH)
    for h5_filename in all_files:
        file_path = provider.getPath(hyper_parameter.DATASET_PATH, h5_filename)
        data_batch, label_batch = provider.loadDataFile(file_path)
        data_batch_list.append(data_batch)
        label_batch_list.append(label_batch)
    data_batches = np.concatenate(data_batch_list, 0)
    label_batches = np.concatenate(label_batch_list, 0)

    # 划分训练集测试集
    room_filelist_path = provider.getPath(hyper_parameter.DATASET_PATH, hyper_parameter.ROOM_FILELIST_PATH)
    room_filelist = provider.getDataFiles(room_filelist_path)
    test_area = 'Area_' + str(test_area)
    train_idxs = []
    test_idxs = []
    for i_, room_name in enumerate(room_filelist):
        if test_area in room_name:
            test_idxs.append(i_)
        else:
            train_idxs.append(i_)

    # 利用列表索引数组
    train_data_ = data_batches[train_idxs, ...]
    train_label_ = label_batches[train_idxs]
    test_data_ = data_batches[test_idxs, ...]
    test_label_ = label_batches[test_idxs]
    train_tiles = train_data_.shape[0]
    test_tiles = test_data_.shape[0]
    train_points = train_data_.shape[0] * train_data_.shape[1]
    test_points = test_data_.shape[0] * test_data_.shape[1]
    print("*-*-*-*-*-*-*-*-Dataset-*-*-*-*-*-*-*-*-*")
    print("*-* Train block:", train_tiles, "Points:", train_points)
    print("*-* Test  block:", test_tiles, " Points:", test_points)
    print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")

    train_data_ = np.expand_dims(train_data_, -1)
    test_data_ = np.expand_dims(test_data_, -1)
    train_label_ = train_label_.astype(np.int32)
    test_label_ = test_label_.astype(np.int32)
    return train_data_.astype(np.float32), train_label_, test_data_.astype(np.float32), test_label_


def constructeModel():
    return model.PointNet(
        hyper_parameter.NUM_POINT,
        hyper_parameter.NUM_ATTRIBUTE,
        hyper_parameter.NUM_CLASSES
    )


def summaryModel(network):
    network.build(
        input_shape=(hyper_parameter.BATCH_SIZE,
                     hyper_parameter.NUM_POINT,
                     hyper_parameter.NUM_ATTRIBUTE,
                     hyper_parameter.NUM_CHANNEL)
    )
    network.summary()


def getLearningRate(staircase=True):
    return tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=hyper_parameter.INITIAL_LEARNING_RATE,
        decay_steps=hyper_parameter.DECAY_STEPS,
        decay_rate=hyper_parameter.DECAY_RATE,
        staircase=staircase
    )


def getOptimizer():
    return tf.optimizers.Adam(
        learning_rate=getLearningRate()
    )


def getDataset(data__, label__):
    return tf.data.Dataset.from_tensor_slices((data__, label__)), data__.shape[0]


def genDatasets(train_data_, train_label_, test_data_, test_label_):
    train_dataset_, train_counts_ = getDataset(train_data_, train_label_)
    train_dataset_ = train_dataset_.shuffle(buffer_size=train_counts_)
    train_dataset_ = train_dataset_.batch(batch_size=hyper_parameter.BATCH_SIZE)
    test_dataset_, test_counts_ = getDataset(test_data_, test_label_)
    test_dataset_ = test_dataset_.batch(batch_size=hyper_parameter.BATCH_SIZE)
    return train_dataset_, train_counts_, test_dataset_, test_counts_


def train(train_data, train_label, test_data, test_label):
    # Data PipLine
    train_dataset, train_counts, test_dataset, test_counts = genDatasets(train_data, train_label, test_data, test_label)
    # Neural Network
    net = constructeModel()
    # Loss Function
    loss_fun = model.PointNetLoss()
    # Optimizer
    optimizer = getOptimizer()
    # Metric
    acc_metric = tf.metrics.SparseCategoricalAccuracy()
    loss_metric = tf.metrics.Mean()

    # Train Step Definition
    def trainOneBatch(batch_data, batch_label):
        with tf.GradientTape() as tape:
            outputs = net(inputs=batch_data, training=True)
            loss_vale = loss_fun(y_true=batch_label, y_pred=outputs)
        gradients = tape.gradient(loss_vale, net.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, net.trainable_variables))
        loss_metric.update_state(values=loss_vale)
        acc_metric.update_state(y_true=batch_label, y_pred=outputs)

    # Test Step Definition
    def testOneBatch(batch_data, batch_label):
        outputs = net(inputs=batch_data, training=False)
        loss_value = loss_fun(y_true=batch_label, y_pred=outputs)
        loss_value = tf.reduce_mean(loss_value)
        acc_value = tf.metrics.sparse_categorical_accuracy(y_pred=outputs, y_true=batch_label)
        acc_value = tf.reduce_mean(acc_value)
        return loss_value, acc_value

    # Train
    print("###***---Training---***###")
    best_loss = [-0.1]
    best_acc = [-0.1]  # Breakpoint refreshes to be implemented
    for epoch in range(hyper_parameter.EPOCHS):
        total_loss = 0
        total_acc = 0
        counts = 0
        for data, label in train_dataset:
            trainOneBatch(data, label)
            total_loss += loss_metric.result()
            total_acc += acc_metric.result()
            counts += 1
        total_loss /= counts
        total_acc /= counts
        print("---EPOCH {}---".format(epoch))
        print("train_loss:{},train_acc:{}".format(total_loss, total_acc))
        loss_metric.reset_states()
        acc_metric.reset_states()

        # Test
        if epoch % hyper_parameter.TEST_FREQUENCE == 0:
            test_loss = 0
            test_acc = 0
            counts_ = 0
            for data_, label_ in test_dataset:
                value_ = testOneBatch(data_, label_)
                test_loss += value_[0]
                test_acc += value_[1]
                counts_ += 1
            test_loss /= counts_
            test_acc /= counts_
            if test_loss >= best_loss[-1]:
                best_loss.append(test_loss)
            if test_acc >= best_acc[-1]:
                net.save_weights(filepath=hyper_parameter.SAVE_PATH, save_format='tf')
                print("test_acc has been promoted from {} to {}, save weights in {}".format(
                    best_acc[-1],
                    test_acc,
                    hyper_parameter.LOGS)
                )
                best_acc.append(test_acc)
            print("test_loss:{},test_acc:{}".format(test_loss, test_acc))
    print("###***---Training End---***###")


def getIoUV1():
    """
    # IoU OA  this part is too slow
    if pred_tensor.shape != test_label.shape:
        raise Exception("ShapeError:pred_tensor.shape != test_label.shape")
    number = [n for n in range(hyper_parameter.NUM_CLASSES)]
    tp = np.zeros(dtype=np.int64, shape=len(number))
    fp = np.zeros(dtype=np.int64, shape=len(number))
    tn = np.zeros(dtype=np.int64, shape=len(number))
    fn = np.zeros(dtype=np.int64, shape=len(number))
    for i in range(test_label.shape[0]):
        for j in range(test_label.shape[1]):
            for k in number:  # 13 kind of situations
                if int(test_label[i,j]) == number[k]:  # enter P
                    if int(test_label[i, j]) - int(pred_tensor[i, j].numpy()) == 0:  # enter T
                        tp[k] += 1
                    else:  # enter F
                        fp[k] += 1
                else:  # enter N
                    if int(test_label[i,j]) - int(pred_tensor[i,j].numpy()) == 0:  # enter T
                        tn[k] += 1
                    else:  # enter F
                        fn[k] += 1
    """


def getColorDict():
    return {
        0:[255,48,48],1:[255,20,147],2:[255,165,0],
        3:[138,43,226],4:[255,218,185],5:[139,0,139],
        6:[139,0,0],7:[0,255,127],8:[0,255,255],
        9:[255,255,0],10:[219,112,147],11:[131,139,139],
        12:[210,180,140]
    }


def evaluate(test_data, test_label, file_path, save_name, isvisualization=True):
    save_name = "Area_" + str(save_name)
    # inference
    net = constructeModel()
    net.load_weights(file_path)
    pred_tensor = []
    for data in test_data:
        data = tf.expand_dims(data, 0)
        pred = net(data, training=False)
        pred = tf.nn.softmax(pred)
        pred = tf.argmax(pred, -1)
        pred_tensor.append(pred)
    del pred, data
    pred_tensor = tf.concat(pred_tensor, 0)

    # IoU mIoU
    iou_list_ = []
    test_label = tf.cast(test_label, dtype=tf.int32)
    pred_tensor = tf.cast(pred_tensor, dtype=tf.int32)

    for i_ in range(hyper_parameter.NUM_CLASSES):
        # mask tensor
        mask_tensor = tf.fill(dims=test_label.shape, value=i_)
        test_sub_tensor = tf.cast(tf.equal(test_label, mask_tensor), tf.int32)
        pred_sub_tensor = tf.cast(tf.equal(pred_tensor, mask_tensor), tf.int32)
        # tp
        tp = tf.reduce_sum(
            tf.multiply(test_sub_tensor, pred_sub_tensor)
        )
        # fp+fn
        fp_fn = tf.reduce_sum(
            tf.abs(test_sub_tensor - pred_sub_tensor)
        )
        iou_list_.append(
            (tp / (tp + fp_fn)).numpy()
        )
    miou_ = tf.reduce_mean(iou_list_)

    # OA
    oa_ = tf.metrics.Accuracy()
    oa_.update_state(y_true=test_label, y_pred=pred_tensor)
    oa_ = oa_.result()
    del mask_tensor,test_sub_tensor,pred_sub_tensor

    # visualization
    if isvisualization:
        test_data = np.squeeze(test_data, axis=3)
        # convert a var from tensor to numpy array
        pred_tensor_numpy = pred_tensor.numpy()
        del pred_tensor
        test_label_numpy = test_label.numpy()
        del test_label
        # rgb xyz slice
        rgb = np.asarray(test_data[::, ::, 3:6:] * 255, dtype=np.int)
        xyz = np.asarray(test_data[::, ::, :3:], dtype=np.float)
        # color dictionary
        color_dict = getColorDict()
        with open(save_name+"_pred.xyz",'w') as fo_pred:
            with open(save_name+"_labeled.xyz", 'w') as fo_labeled:
                with open(save_name + "_original.xyz", 'w') as fo_orig:
                    for i in range(test_data.shape[0]):
                        for j in range(test_data.shape[1]):
                            # write original rgb
                            fo_orig.write(
                                str(round(xyz[i, j, 0], 3)) + " " +
                                str(round(xyz[i, j, 1], 3)) + " " +
                                str(round(xyz[i, j, 2], 3)) + " " +
                                str(rgb[i, j, 0]) + " " +
                                str(rgb[i, j, 1]) + " " +
                                str(rgb[i, j, 2]) + " " +
                                str(test_label_numpy[i, j]) + "\n"
                            )
                            # write labeled rgb with a color dictionary
                            fo_labeled.write(
                                str(round(xyz[i, j, 0], 3)) + " " +
                                str(round(xyz[i, j, 1], 3)) + " " +
                                str(round(xyz[i, j, 2], 3)) + " " +
                                str(color_dict[test_label_numpy[i, j]][0]) + " " +
                                str(color_dict[test_label_numpy[i, j]][1]) + " " +
                                str(color_dict[test_label_numpy[i, j]][2]) + " " +
                                str(test_label_numpy[i, j]) + "\n"
                            )
                            # write predicted rgb with a color dictionary
                            fo_pred.write(
                                str(round(xyz[i, j, 0], 3)) + " " +
                                str(round(xyz[i, j, 1], 3)) + " " +
                                str(round(xyz[i, j, 2], 3)) + " " +
                                str(color_dict[pred_tensor_numpy[i, j]][0]) + " " +
                                str(color_dict[pred_tensor_numpy[i, j]][1]) + " " +
                                str(color_dict[pred_tensor_numpy[i, j]][2]) + " " +
                                str(pred_tensor_numpy[i, j]) + "\n"
                            )

    with open("evaluation.txt","a") as fo_eval:
        fo_eval.write(
            save_name + '\n' +
            "mIoU:{}\nOA:{}\nIoU_list:{}\n".format(miou_.numpy(),oa_.numpy(),iou_list_)
        )
    return miou_.numpy(), oa_.numpy(), iou_list_


def kFoldCrossVal(k_fold):
    # load_data
    train_point, train_labels, test_point, test_labels = load_data(k_fold)
    train(train_data=train_point,
          train_label=train_labels,
          test_data=test_point,
          test_label=test_labels)
    miou, oa, iou_list = evaluate(test_data=test_point,
                                  test_label=test_labels,
                                  file_path=hyper_parameter.SAVE_PATH,
                                  save_name=k_fold)
    return miou, oa, iou_list


if __name__ == '__main__':
    """
     # k-fold
    # remove old files generated by last running
    del_files = provider.match_postfix("xyz", iscwd=True)
    provider.removeFiles(del_files)
    del_files = provider.match_postfix("txt", iscwd=True)
    provider.removeFiles(del_files)
    for i_fold in range(1,3):
        result = kFoldCrossVal(i_fold)
    """
    a,b,c,d = load_data()
    print()

