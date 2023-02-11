import time
import os
import configparser
from keras.callbacks import ReduceLROnPlateau
import numpy as np
from data_processed import data_generator
from datetime import datetime
import keras
from config import get_configuration
from build_model import build_lstm_model

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# 保存所有超参数至单独的txt文件（字典）
# 完成数据读取部分的代码优化
# 完成模型部分建立的优化

# 基本参数设置
# wave:0, surge:1, sway:2, heave:3, roll:4, pitch:5, yaw:6
window, lag, input_index, output_index = get_configuration()


# 训练数据集准备
proDir = os.path.split(os.path.realpath(__file__))[0]
dataPath = os.path.join(proDir, "new_data")
resultPath = os.path.join(proDir, 'results')


# nb_data = 27
# for i in range(nb_data):
#     # W1: 9, W2: 8, W3: 6, W4: 4
#     if i < 9:
#         locals()['data' + str(i)] = np.loadtxt(os.path.join(dataPath, "W1", str(i + 1) + '.dat'))
#     elif i < 17:
#         locals()['data' + str(i)] = np.loadtxt(os.path.join(dataPath, "W2", str(i - 8) + '.dat'))
#     elif i < 23:
#         locals()['data' + str(i)] = np.loadtxt(os.path.join(dataPath, "W3", str(i - 16) + '.dat'))
#     else:
#         locals()['data' + str(i)] = np.loadtxt(os.path.join(dataPath, "W4", str(i - 22) + '.dat'))

nb_data = 12
for i in range(nb_data):
    # W1: 9, W2: 8, W3: 6, W4: 4
    if i < 3:
        locals()['data' + str(i)] = np.loadtxt(os.path.join(dataPath, "W1", str(i + 1) + '.dat'))
    elif i < 6:
        locals()['data' + str(i)] = np.loadtxt(os.path.join(dataPath, "W2", str(i - 2) + '.dat'))
    elif i < 9:
        locals()['data' + str(i)] = np.loadtxt(os.path.join(dataPath, "W3", str(i - 5) + '.dat'))
    else:
        locals()['data' + str(i)] = np.loadtxt(os.path.join(dataPath, "W4", str(i - 8) + '.dat'))

for j in range(nb_data):
    locals()['x_train' + str(j)], locals()['y_train' + str(j)] = data_generator(locals()['data' + str(j)],
                                                                                window=window,
                                                                                lag=lag,
                                                                                input_index=input_index,
                                                                                output_index=output_index)
    del locals()['data' + str(j)]

x_train = x_train0
y_train = y_train0
for j in range(1, nb_data):
    x_train = np.concatenate([x_train, locals()['x_train' + str(j)]], axis=0)
    y_train = np.concatenate([y_train, locals()['y_train' + str(j)]], axis=0)
    del locals()['x_train' + str(j)], locals()['y_train' + str(j)]

# 测试数据集准备
test_data = np.loadtxt(os.path.join(dataPath, "W4", '4.dat'))
x_test, y_test = data_generator(test_data, window=window, lag=lag, input_index=input_index, output_index=output_index)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

state = np.random.get_state()
np.random.shuffle(x_train)
np.random.set_state(state)
np.random.shuffle(y_train)


def run_task(model=None, epoch=0):
    if model is None:
        model = build_lstm_model(x_train.shape[1], x_train.shape[2])
        print("We will build the model")
    else:
        model = keras.models.load_model(model)
        print("We will load the model")
    start_time = time.time()

    if epoch > 0:
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=4, mode='auto', factor=0.1)
        model.fit(x_train, y_train, batch_size=256, validation_split=0.1, epochs=epoch, callbacks=[reduce_lr])
        name = 'heave'
        model.save(name + '.h5')
        print('Model saved!')
    end_time = time.time()

    predicted = np.array(model.predict(x_test))
    predicted = np.reshape(predicted, (predicted.size,))

    time_now = datetime.now()
    time_label = str(time_now.month) + str(time_now.day) + str(time_now.hour) + str(time_now.minute)
    time_label = time_label.zfill(8)
    # end_path = "%s/" % resultPath
    end_path = os.path.join(resultPath, time_label)
    os.makedirs(end_path)

    predict_save_path = "{}/{}.txt".format(end_path, 'predict')
    y_test_save_path = "{}/{}.txt".format(end_path, 'y_test')
    # x_test_save_path = "{}/{}.txt".format(end_path, 'x_test')
    np.savetxt(predict_save_path, predicted, fmt='%.4f', delimiter=',')
    np.savetxt(y_test_save_path, y_test, fmt='%.4f', delimiter=',')
    # np.savetxt(x_test_save_path, x_test, fmt='%.4f', delimiter=',')


if __name__ == '__main__':
    old_time = time.time()
    run_task(model=None, epoch=100)
    current_time = time.time()
    print('time is: ', str(current_time - old_time) + 's')
