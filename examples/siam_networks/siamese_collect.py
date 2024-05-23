import glob
import numpy as np
import matplotlib.pyplot as plt

MNIST_weights_train = np.zeros((60000, 60000), dtype=np.float32)

files = glob.glob("siamese_MNIST_train_set_*")

for file in files:
    position = int(file.split("_")[4])
    w_part = np.load(file)
    MNIST_weights_train[position:position + 1000] = w_part

np.save("D:\siamese_MNIST_train_set.npy", MNIST_weights_train)

plt.imshow(MNIST_weights_train, vmax=0.1)
plt.colorbar()
plt.show()
