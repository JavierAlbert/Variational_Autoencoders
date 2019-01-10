from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import numpy as np

# Load data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Reshape images to 784 pixels and stack them
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print (x_train.shape)
print (x_test.shape)

input_img = Input(shape=(28, 28, 1))

# Define encoding layers
encode = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
encode = MaxPooling2D((2, 2), padding='same')(encode)
encode = Conv2D(8, (3, 3), activation='relu', padding='same')(encode)
encode = MaxPooling2D((2, 2), padding='same')(encode)
encode = Conv2D(8, (3, 3), activation='relu', padding='same')(encode)
encode = MaxPooling2D((2, 2), padding='same')(encode)

# Define decoding layers
decode = Conv2D(8, (3, 3), activation='relu', padding='same')(encode)
decode = UpSampling2D((2, 2))(decode)
decode = Conv2D(8, (3, 3), activation='relu', padding='same')(decode)
decode = UpSampling2D((2, 2))(decode)
decode = Conv2D(16, (3, 3), activation='relu')(decode)
decode = UpSampling2D((2, 2))(decode)
decode = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(decode)

# Define Model
autoencoder = Model(input_img, decode)

# Compile Model
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# Let's take a peak at the architecture
autoencoder.summary()

# Fit autoencoder
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                verbose=0)

# Let's use the autoencoder on some of the test images
decoded_imgs = autoencoder.predict(x_test)

# Now we can take a look
plt.figure(figsize=(20, 4))
for i in range(10):
    # display original
    ax = plt.subplot(2, 10, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, 10, i + 1 + 10)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()