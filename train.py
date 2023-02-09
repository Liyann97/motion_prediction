import time
import os
from keras.callbacks import ReduceLROnPlateau
import numpy as np
from data_processed import data_generator
import keras

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# 基本参数设置
# wave:0, surge:1, sway:2, heave:3, roll:4, pitch:5, yaw:6
window = 100
lag = 50
input_index = 0
output_index = 3

# 训练数据集准备
nb_data = 27
for i in range(nb_data):
    # W1: 9, W2: 8, W3: 6, W4: 4
    if i < 9:
        locals()['data' + str(i)] = np.loadtxt(r'F:/my_data/W1/'+str(i + 1)+'.dat')
    elif i < 17:
        locals()['data' + str(i)] = np.loadtxt(r'F:/my_data/W2/' + str(i - 8) + '.dat')
    elif i < 23:
        locals()['data' + str(i)] = np.loadtxt(r'F:/my_data/W3/' + str(i - 16) + '.dat')
    else:
        locals()['data' + str(i)] = np.loadtxt(r'F:/my_data/W4/' + str(i - 22) + '.dat')

for j in range(nb_data):
    locals()['x_train'+str(j)], locals()['y_train'+str(j)] = data_generator(locals()['data'+str(j)],
                                                                            window=window,
                                                                            lag=lag,
                                                                            input_index=input_index,
                                                                            output_index=output_index)
    del locals()['data'+str(j)]

x_train = x_train0
y_train = y_train0
for j in range(1, nb_data):
    x_train = np.concatenate([x_train, locals()['x_train' + str(j)]], axis=0)
    y_train = np.concatenate([y_train, locals()['y_train' + str(j)]], axis=0)
    del locals()['x_train'+str(j)], locals()['y_train'+str(j)]

# 测试数据集准备
test_data = np.loadtxt(r'F:/my_data/W1/10hz/W1_4a.dat')
x_test, y_test = data_generator(test_data, window=window, lag=lag, input_index=input_index, output_index=output_index)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

state = np.random.get_state()
np.random.shuffle(x_train)
np.random.set_state(state)
np.random.shuffle(y_train)


def build_lstm_model():
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(128, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=False))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(1, activation='linear'))

    model.compile(loss='mse', optimizer='adam')
    keras.optimizers.Adam(lr=0.0001)

    return model


def run_task(model=None, epoch=0):
    if model is None:
        model = build_lstm_model()
        print("We will build the model")
    else:
        model = keras.models.load_model(model)
        print("We will load the model")

    if epoch > 0:
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=4, mode='auto', factor=0.1)
        model.fit(x_train, y_train, batch_size=256, validation_split=0.1, epochs=epoch, callbacks=[reduce_lr])
        name = 'heave'
        model.save(name + '.h5')
        print('Model saved!')

    predicted = np.array(model.predict(x_test))
    predicted = np.reshape(predicted, (predicted.size,))

    np.savetxt('predict.txt', predicted, fmt='%.4f', delimiter=',')
    np.savetxt('y_test.txt', y_test, fmt='%.4f', delimiter=',')
    # np.savetxt('x_test.txt', x_test, fmt='%.4f', delimiter=',')


if __name__ == '__main__':
    old_time = time.time()
    run_task(model=None, epoch=100)
    current_time = time.time()
    print('time is: ', str(current_time - old_time) + 's')