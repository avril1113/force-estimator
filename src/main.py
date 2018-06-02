import numpy as np
from keras import losses
from keras.models import Model
from keras.layers import Input, Dense, Dropout, GRU, Flatten
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split


def create_model(weights_path=None):
    # custom layers
    inputs = Input(shape=(1099, 9))
    net = GRU(32, return_sequences=True)(inputs)
    net = Dropout(0.5)(net)
    net = GRU(32, return_sequences=True)(net)
    net = Dropout(0.5)(net)
    net = GRU(32, return_sequences=True)(net)
    net = Dropout(0.25)(net)
    net = GRU(16, return_sequences=True)(net)
    net = Dropout(0.25)(net)
    net = Flatten()(net)
    net = Dense(1099)(net)
    net = Model(inputs=inputs, outputs=net)

    # set up optimizer
    optimizer = RMSprop(lr=0.001)

    net.compile(loss=losses.mean_squared_error, optimizer=optimizer)
    print (net.summary())

    if weights_path:
        net.load_weights(weights_path)

    return net


if __name__ == "__main__":
    data_path = 'train.npy'

    # extract data
    data = np.load(data_path)
    f_t = data[:, :, 1]
    train_data = np.delete(data, [1], 2)

    # split data to training, validation, and testing
    training_data, testing_data, training_f_data, testing_f_data = train_test_split(train_data, f_t, test_size=0.25)

    model = create_model()
    model.fit(training_data, training_f_data, epochs=200, batch_size=64)
    result = model.evaluate(testing_data, testing_f_data)
    print (result)

    # save the model
    model.save('model.h5')
