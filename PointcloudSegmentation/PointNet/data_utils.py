"""
    It is apologetic for some ambiguous arguments and redundancy in few implement due to my limitations, but I always
this part can help construct your point cloud dataset fastly and imperative maintenance must be provided.
"""
import numpy as np
import provider
import h5py
import os

# Point Cloud Path
POINT_CLOUD_PATH = os.path.join(os.getcwd(),"data\\original")  # please change the path to your point cloud

# Zoom Factor
ZOOM_FACTOR = 0.048314  # my points, which is derive from 3dmax model, needed to be zoom as the real size.


def getColorDict():
    return provider.getColorDict()


def zoomPointCloud(cloud, factor):
    """
    zoom point cloud with a factor
    :param cloud: numpy array
    :param factor: zoom factor
    :return: np.array of float32
    """
    if not isinstance(cloud, type(np.array(0))):
        raise Exception("inputs must be np.ndarray instead {}.".format(type(cloud)))
    return np.array(cloud * factor, dtype=np.float32)


def shiftPointCloud(cloud):
    """
    :param cloud:numpy ndarray
    :return: np.ndarray with shited point cloud
    """
    if not isinstance(cloud, type(np.array(0))):
        raise Exception("inputs must be np.ndarray instead {}.".format(type(cloud)))
    xyz_min = np.amin(cloud, axis=0)[0:3]
    cloud[:, 0:3] -= xyz_min
    return cloud


def readText(path, factor=1.0, iszoom=True):
    """ load text file
        data info  float32 [room1,room2,...] shape:[N×3,N×3,...] attributes: x y z
        label info uint8   [label1,lable2,...] shape:[N×1,N×1,...] attributes: label
        room info  string  [name1,name2,...]
    :param path: string, the root path of point cloud file
    :param factor: if 'iszoom' is true, factor will be multiple coordinates
    :param iszoom: bool, if true, will zoom point cloud
    :return: three list of data, label and name of these rooms
    """

    point_cloud_text = provider.match_postfix('txt', path)
    data_ = []  # N×3
    label_ = []  # N
    room_ = []  # N
    for text_ in point_cloud_text:
        file_name_ = text_.split(provider.getPathSeparator())[-1]
        # read all lines
        with open(text_) as fi:
            data_label_ = np.array(
                [j_.split(" ") for j_ in [i_.strip() for i_ in fi.readlines()]],
                dtype=np.float32
            )
        data_.append(data_label_[:, :-1])
        label_.append(np.array(data_label_[:, -1], np.uint8))
        room_.append(file_name_[:-4])
    del data_label_
    # Zoom Point Cloud
    if iszoom:
        data_ = [zoomPointCloud(data_points, factor) for data_points in data_]

    # Shift Point Cloud
    data_ = [shiftPointCloud(data_points) for data_points in data_]
    return data_, label_, room_


def randomSamplePoints(points, label, name, num_sample=4096):
    """
    The shape of points is N×3, x y z. we want to sample all of the points until it is exhausted.
    :param points: np.ndarray
    :param label: np.ndarray
    :param name: string
    :param num_sample: int, point numbers of each block
    :return: np.ndarray, batch of points
    """
    # Check type
    if ".txt" in name:
        name = name.replace(".txt", "")
    if not (isinstance(points, type(np.array(0))) & isinstance(label, type(np.array(0)))):
        raise Exception("values must be np.ndarray.")

    batch_data = []
    batch_label = []
    batch_name = []

    point_id = list(range(points.shape[0]))  # record every point
    batch_id = 0
    while len(point_id) > 0:
        np.random.shuffle(point_id)
        if len(point_id) > num_sample:
            sample = point_id[:num_sample]
        elif len(point_id) == num_sample:
            sample = point_id[:num_sample]
        else:
            sample = np.random.choice(point_id, num_sample, True)
        batch_data.append(np.expand_dims(points[sample, ::], 0))
        batch_label.append(np.expand_dims(label[sample], 0))
        batch_name.append(name + '_batch' + str(batch_id))
        batch_id += 1
        del point_id[:num_sample]
    batch_data = np.concatenate(batch_data, 0)  # B×N×3
    batch_label = np.concatenate(batch_label, 0)  # B×N
    return batch_data, batch_label, batch_name


def roomFixedBlock(data, label, name, stride, block_size):
    """
    Prepare block training data.
    :param data: N×3 numpy array
    :param label: N×1 numpy array
    :param name: string, the name of a room
    :param stride: int
    :param block_size: int
    :return: np.ndarray np.ndarray list
    """
    if not (isinstance(data, np.ndarray) & isinstance(label, np.ndarray) & isinstance(name, str)):
        raise Exception("values error.")
    assert (stride <= block_size)
    if ".txt" in name:
        name = name.replace(".txt", "")

    # Point Cloud Border
    limit = np.amax(data, 0)
    # Compute the Corner Coordinates
    xbeg_list = []
    ybeg_list = []
    num_block_x = int(np.ceil((limit[0] - block_size) / stride)) + 1
    num_block_y = int(np.ceil((limit[1] - block_size) / stride)) + 1
    for i_ in range(num_block_x):
        xbeg_list.append(i_ * stride)
    for j_ in range(num_block_y):
        ybeg_list.append(j_ * stride)

    # Block the whole Room, which is shifted and aligned with a real size
    block_data_list = []
    block_label_list = []
    block_name_list = []
    block_id = 0
    for i_ in range(len(xbeg_list)):
        for j_ in range(len(ybeg_list)):
            xbeg = xbeg_list[i_]
            ybeg = ybeg_list[j_]
            # numpy array mask trick using broadcast and indexing with logistic value
            xcond = (data[:, 0] <= xbeg + block_size) & (data[:, 0] >= xbeg)
            ycond = (data[:, 1] <= ybeg + block_size) & (data[:, 1] >= ybeg)
            cond = xcond & ycond
            if np.sum(cond) < 100:  # discard block if there are less than 100 pts.
                continue
            block_data_list.append(data[cond, :])
            block_label_list.append(label[cond])
            block_name_list.append(name + "_block" + str(block_id))
            block_id += 1

    # Sample points of each block
    sample_batch_data = []
    sample_batch_label = []
    batch_name_ = []
    for idx in range(block_id):
        sample_data, sample_label, sample_name = randomSamplePoints(
            points=block_data_list[idx],
            label=block_label_list[idx],
            name=block_name_list[idx]
        )
        sample_batch_data.append(sample_data)
        sample_batch_label.append(sample_label)
        batch_name_.extend(sample_name)
    del sample_data, sample_label, sample_name
    batch_data_ = np.concatenate(sample_batch_data, 0)
    del sample_batch_data
    batch_label_ = np.concatenate(sample_batch_label, 0)
    del sample_batch_label

    return batch_data_, batch_label_, batch_name_


def room2BlockNormalized(data_batch, block_size):
    """
    Normalized points of room for each clock.
    :param data_batch: array, B×N×3
    :param block_size: int
    :return: numpy n-d array, B×N×6:x y z X Y Z
    """
    if not (isinstance(data_batch, np.ndarray)):
        raise Exception("values error.")

    max_room_x = np.max(data_batch[:, :, 0])
    max_room_y = np.max(data_batch[:, :, 1])
    max_room_z = np.max(data_batch[:, :, 2])

    new_data_batch = np.zeros((data_batch.shape[0], data_batch.shape[1], 6))
    for idx in range(data_batch.shape[0]):
        new_data_batch[idx, :, 3] = data_batch[idx, :, 0] / max_room_x
        new_data_batch[idx, :, 4] = data_batch[idx, :, 1] / max_room_y
        new_data_batch[idx, :, 5] = data_batch[idx, :, 2] / max_room_z
        minx = np.min(data_batch[idx, :, 0])
        miny = np.min(data_batch[idx, :, 1])
        data_batch[idx, :, 0] -= (minx + block_size / 2)
        data_batch[idx, :, 1] -= (miny + block_size / 2)
    new_data_batch[:, :, :3] = data_batch[:, :, :3]

    return new_data_batch


def rooms2BlockNormalized(data_list,label_list,name_list,
                          stride=1.0,block_size=1.0,
                          save_block=False,save_normalized=False,record_blocked_coord=False):
    if not (isinstance(data_list, list) & isinstance(label_list, list) & isinstance(name_list, list)):
        raise Exception("values must be list.")

    if len(data_list) != len(label_list) != name_list:
        raise Exception("Data error.")

    # Block
    batch_data_list = []
    batch_label_list = []
    batch_name = []
    for idx in range(len(name_list)):
        batch_data_, batch_label_, batch_name_ = roomFixedBlock(
            data=data_list[idx],
            label=label_list[idx],
            name=name_list[idx],
            stride=stride,
            block_size=block_size
        )
        batch_data_list.append(batch_data_)
        batch_label_list.append(batch_label_)
        batch_name.extend(batch_name_)
    del batch_data_, batch_label_, batch_name_

    if save_block:
        # Merge all block points to check code of blocking and shifting rooms
        color_dict = getColorDict()
        for idx in range(len(name_list)):
            with open(name_list[idx]+"_blocked.txt",'w') as fo:
                for i_ in range(batch_data_list[idx].shape[0]):
                    for j_ in range(batch_data_list[idx].shape[1]):
                        point_label = int(batch_label_list[idx][i_, j_])
                        fo.write(
                            str(batch_data_list[idx][i_, j_, 0]) + " " +
                            str(batch_data_list[idx][i_, j_, 1]) + " " +
                            str(batch_data_list[idx][i_, j_, 2]) + " " +
                            str(color_dict[point_label][0]) + " " +
                            str(color_dict[point_label][1]) + " " +
                            str(color_dict[point_label][2]) + " " +
                            str(point_label) + "\n"
                            )
    if record_blocked_coord:
        for idx in range(len(name_list)):
            with h5py.File(name_list[idx]+"_record_blocked_coordinates.h5",'w') as fo_h5:
                fo_h5['data'] = batch_data_list[idx]
                fo_h5['label'] = batch_label_list[idx]
    batch_data = np.concatenate(batch_data_list, 0)
    del batch_data_list
    batch_label = np.concatenate(batch_label_list, 0)
    del batch_label_list

    # Normalized
    batch_data = room2BlockNormalized(batch_data, block_size)  # B×N×6

    if save_normalized:
        # Save normalized points to check code of normalizing
        color_dict = getColorDict()
        for room_name in name_list:
            block_id = []
            for id_,block_name in enumerate(batch_name):
                if room_name in block_name:
                    block_id.append(id_)
            point_cloud = batch_data[block_id,:,:3]
            point_labels = batch_label[block_id,:]
            with open(room_name+"_normalized.txt","w") as fo:
                for i_ in range(point_cloud.shape[0]):
                    for j_ in range(point_cloud.shape[1]):
                        point_label = point_labels[i_,j_]
                        fo.write(
                            str(point_cloud[i_,j_,0]) + " " +
                            str(point_cloud[i_,j_,1]) + " " +
                            str(point_cloud[i_,j_,2]) + " " +
                            str(color_dict[point_label][0]) + " " +
                            str(color_dict[point_label][1]) + " " +
                            str(color_dict[point_label][2]) + " " +
                            str(point_label) + "\n"
                        )

    return batch_data, batch_label, batch_name


def generateData(path,save_name,zoom_factor=1.0,iszoom=True,stride=1.0,block_size=1.0,
                 save_block=False,save_normalized=False,record_blocked_coord=False):
    """
    x y z label -> x y z X Y Z and label
    txt -> hdf5
    original coordinates -> normalized coordinates and relative coordinates
    :param path: original files
    :param save_name: name of generated h5 file
    :param zoom_factor: default 1.0
    :param iszoom: bool, if True the zoom_factor and original coordinates will be multiply
    :param stride: sample stride, default 1.0
    :param block_size: size of sample block, default 1.0
    :param save_block: bool, if you want to save the points after blocked, it will be True
    :param save_normalized:bool, save normalized points
    :param record_blocked_coord:bool, record original coordinates as the order of blocked points in order to visualize
    :return: None
    """
    data_batch_list, label_batch_list, room_name_list = readText(path, zoom_factor, iszoom)
    data_batch,label_batch,name_batch_list = rooms2BlockNormalized(
        data_list=data_batch_list,
        label_list=label_batch_list,
        name_list=room_name_list,
        stride=stride,
        block_size=block_size,
        save_block=save_block,
        save_normalized=save_normalized,
        record_blocked_coord=record_blocked_coord
    )

    with h5py.File(save_name+".h5",'w') as h5_fo:
        h5_fo['data'] = data_batch
        h5_fo['label'] = label_batch
        with open("all_batch_names.txt",'w') as txt_fo:
            for batch_name in name_batch_list:
                txt_fo.write(batch_name+'\n')


if __name__ == "__main__":
    # generate h5 file and a txt recorded each name of batch
    generateData(
        path=POINT_CLOUD_PATH,
        save_name="ArchitectureData",
        zoom_factor=ZOOM_FACTOR,
        stride=1.0,
        block_size=1.0,
        save_block=True,
        save_normalized=True,
        record_blocked_coord=True
    )




