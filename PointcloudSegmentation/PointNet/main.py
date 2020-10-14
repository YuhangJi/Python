import hyper_parameter
import provider
import model

import datetime
import warnings
import numpy as np
import tensorflow as tf


def load_data(test_area):
    # 读取点云数据
    data_batch_list = []
    label_batch_list = []
    all_files = provider.getDataFiles(hyper_parameter.ALL_FILES_PATH)
    for h5_filename in all_files:
        file_path = provider.getPath(hyper_parameter.DATASET_PATH, h5_filename)
        data_batch, label_batch = provider.loadDataFile(file_path)
        data_batch_list.append(data_batch)
        label_batch_list.append(label_batch)
    del data_batch,label_batch
    data_batches = np.concatenate(data_batch_list, 0)
    label_batches = np.concatenate(label_batch_list, 0)

    # 划分训练集测试集
    room_filelist_path = provider.getPath(hyper_parameter.DATASET_PATH, hyper_parameter.ROOM_FILELIST_PATH)
    room_filelist = provider.getDataFiles(room_filelist_path)
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
    train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')
    test_loss_metric = tf.keras.metrics.Mean(name='test_loss')
    test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_acc')

    # Train Step Definition
    def trainOneBatch(batch_data, batch_label):
        with tf.GradientTape() as tape:
            outputs = net(inputs=batch_data, training=True)
            train_loss_vale = loss_fun(y_true=batch_label, y_pred=outputs)
        gradients = tape.gradient(train_loss_vale, net.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, net.trainable_variables))
        train_loss_metric.update_state(values=train_loss_vale)
        train_acc_metric.update_state(y_true=batch_label, y_pred=outputs)

    # Test Step Definition
    def testOneBatch(batch_data, batch_label):
        outputs = net(inputs=batch_data, training=False)
        test_loss_value = loss_fun(y_true=batch_label, y_pred=outputs)
        test_loss_metric.update_state(values=test_loss_value)
        test_acc_metric.update_state(y_true=batch_label,y_pred=outputs)

    # LOGS Tracing
    stamp = datetime.datetime.now().strftime("Y%m%d-%H%M%S")
    log_dir = hyper_parameter.LOGS_DIR(hyper_parameter.LOGS,stamp)
    summary_writer = tf.summary.create_file_writer(log_dir)
    tf.summary.trace_on(graph=True, profiler=False)

    # Train Model
    print("###***---Training---***###")
    best_loss = [float('inf')]
    best_acc = [float('-inf')]  # Breakpoint refreshes to be implemented
    for epoch in range(hyper_parameter.EPOCHS):
        # Reset Metric
        train_loss_metric.reset_states()
        train_acc_metric.reset_states()
        test_loss_metric.reset_states()
        test_acc_metric.reset_states()

        for data, label in train_dataset:
            trainOneBatch(data, label)

        print("---EPOCH {}---".format(epoch))
        print("train_loss:{},train_acc:{}".format(train_loss_metric.result(), train_acc_metric.result()))

        with summary_writer.as_default():
            tf.summary.scalar('train_loss', train_loss_metric.result(), step=epoch)
            tf.summary.scalar('train_accuracy', train_acc_metric.result(), step=epoch)

        # Test
        if epoch % hyper_parameter.TEST_FREQUENCE == 0:
            for data_, label_ in test_dataset:
                testOneBatch(data_, label_)

            if test_loss_metric.result() < best_loss[-1]:
                best_loss.append(test_loss_metric.result())

            if test_acc_metric.result() > best_acc[-1]:
                print("test_loss:{},test_acc:{}".format(test_loss_metric.result(), test_acc_metric.result()))
                best_acc.append(test_acc_metric.result())
                net.save_weights(filepath=hyper_parameter.SAVE_PATH, save_format='tf')
                print("test_acc has been promoted from {} to {}, save weights in {}".format(
                    best_acc[-2],
                    best_acc[-1],
                    hyper_parameter.LOGS)
                )
            else:
                print("test_loss:{},test_acc:{}".format(test_loss_metric.result(), test_acc_metric.result()))

            with summary_writer.as_default():
                tf.summary.scalar('test_loss', test_loss_metric.result(), step=epoch)
                tf.summary.scalar('test_accuracy', test_acc_metric.result(), step=epoch)

    # Stop Tracing and Export
    with summary_writer.as_default():
        tf.summary.trace_export(name="model_trace",step=3)

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
    pass


def evaluate(test_data, test_label, file_path,name, isvisualization=False):
    """
    :param test_data: numpy array
    :param test_label: numpy array
    :param file_path: model weight path
    :param name: string, room name
    :param isvisualization: bool, if true, it will generate two point cloud where one is labeled and another is evaluated
    :return: miou, oa and iou_list
    """
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
    color_dict = provider.getColorDict()
    if isvisualization:
        # convert a var from tensor to numpy array
        pred_label_numpy = pred_tensor.numpy()
        del pred_tensor
        test_label_numpy = test_label.numpy()
        del test_label
        original_coordinates,original_label = provider.getOriginalCoordinates(
            name=name,
            file_path=hyper_parameter.ORIGINAL_COORDINATES_PATH
        )

        if int(np.sum(original_label - test_label_numpy)) != 0:
            warnings.warn("Original coordinates is inconsistent with evaluated points")

        with open(name + "_predicted.txt", 'w') as fo_pre:
            with open(name + "_labeled.txt", 'w') as fo_lab:
                for i_ in range(original_coordinates.shape[0]):
                    for j_ in range(original_coordinates.shape[1]):
                        pred_result = int(pred_label_numpy[i_, j_])
                        label_result = int(test_label_numpy[i_, j_])
                        fo_pre.write(
                            str(original_coordinates[i_, j_, 0]) + " " +
                            str(original_coordinates[i_, j_, 1]) + " " +
                            str(original_coordinates[i_, j_, 2]) + " " +
                            str(color_dict[pred_result][0]) + " " +
                            str(color_dict[pred_result][1]) + " " +
                            str(color_dict[pred_result][2]) + " " +
                            str(pred_result) + "\n"
                        )
                        fo_lab.write(
                            str(original_coordinates[i_, j_, 0]) + " " +
                            str(original_coordinates[i_, j_, 1]) + " " +
                            str(original_coordinates[i_, j_, 2]) + " " +
                            str(color_dict[label_result][0]) + " " +
                            str(color_dict[label_result][1]) + " " +
                            str(color_dict[label_result][2]) + " " +
                            str(label_result) + "\n"
                        )

    with open("evaluation.txt","a") as fo_eval:
        fo_eval.write(
            name + '\n' +
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
                                  name=k_fold,
                                  isvisualization=True)
    return miou, oa, iou_list


if __name__ == '__main__':
    k_fold_list = ["li_jing_xuan", "tai_ji_dian", "tong_dao_tang", "yan_qi_men", "zhong_you_men"]
    # # k-fold
    # for i_fold in k_fold_list:
    #     result = kFoldCrossVal(i_fold)

    k = 4
    train_data_batch,train_label_batch,test_data_batch,test_label_batch = load_data(k_fold_list[k])
    train(train_data_batch,train_label_batch,test_data_batch,test_label_batch)
    evaluate(test_data_batch,test_label_batch,hyper_parameter.SAVE_PATH,k_fold_list[k],True)




