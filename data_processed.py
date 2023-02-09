import sys

import numpy as np
from numpy import array


# 数据处理：
# # 删除多余通道: 0:time, 1:incident wave, 2~7:motions, 8~33:air-gap, 10:AG3
# # 下采样：降低数据量，避免因相位误差导致的label波动
# # 数据切片：根据时间关系，构建X-Y对
def delete_feature(data_series):
    """
    The incident wave, 6-DOF motions (surge, sway, heave, roll, pitch, yaw), and AG3 were kept
    :param data_series:
    :return:
    """
    keep_index = [1, 2, 3, 4, 5, 6, 7]
    series_cut = data_series[:, keep_index]
    return series_cut


def down_sample(data_series, frq_origin, frq_processed):
    if frq_origin % frq_processed != 0 or frq_origin <= frq_processed:
        print("The frequency cannot be downsampled! ")
        sys.exit(1)
    else:
        skip_step = frq_origin / frq_processed
        delete_index = []
        for i in range(data_series.shape[0]):
            if i % skip_step != 0:
                delete_index.append(i)
        series_processed = np.delete(data_series, delete_index, axis=0)
    return series_processed


def slice2pair(data_series, window, lag, input_index, output_index):
    length = data_series.shape[0]
    x_dataset = []
    y_dataset = []
    for i in range(length - lag - window):
        x_dataset.append(data_series[i:i + window + lag, input_index])
    for j in range(window - 1, length - lag - 1):
        y_dataset.append(data_series[j, output_index])

    x = np.array(x_dataset)
    y = np.array(y_dataset)
    return x, y


def data_generator(data_series, window, lag, input_index, output_index):
    dataset = delete_feature(data_series)
    dataset = down_sample(dataset, frq_origin=10, frq_processed=5)
    x_data, y_data = slice2pair(dataset, window=window, lag=lag, input_index=input_index, output_index=output_index)
    x_data = np.expand_dims(x_data, axis=2)
    y_data = np.expand_dims(y_data, axis=1)
    return x_data, y_data


