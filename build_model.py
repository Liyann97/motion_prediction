import keras


def build_lstm_model(input_x_shape, input_y_shape):
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(128, input_shape=(input_x_shape, input_y_shape), return_sequences=False))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(1, activation='linear'))

    model.compile(loss='mse', optimizer='adam')
    keras.optimizers.Adam(lr=0.0001)

    return model

def build_tcn_model():

    return model
