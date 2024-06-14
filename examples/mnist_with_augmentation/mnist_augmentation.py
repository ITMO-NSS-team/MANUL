import numpy as np
import cv2
import torchvision


mnist = torchvision.datasets.MNIST('../data/', download=True)
X_train = mnist.train_data.numpy()
Y_train = mnist.train_labels.numpy()

unique_classes = np.unique(Y_train)


def create_dataset(x_data, y_data):
    x_result = []
    y_result = []
    ang_result = []

    def append_data(newx, newy, ang):
        x_result.append(newx)
        y_result.append(newy)
        ang_result.append(ang)

    for i in range(x_data.shape[0]):
        image = x_data[i]
        label = y_data[i]
        append_data(image, label, 0)

        height, width = image.shape[:2]

        for ange in range(15, 180, 30):
            rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), ange, 1)
            rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
            append_data(rotated_image, label, ange)
    return x_result, y_result, ang_result


new_train_dataX, new_train_dataY, new_train_dataAng = create_dataset(X_train, Y_train)

new_train_dataX = np.array(new_train_dataX)
new_train_dataY = np.array(new_train_dataY)
new_train_dataAng = np.array(new_train_dataAng)

np.save("../data/feature_mnist.npy", new_train_dataX)
np.save("../data/target_mnist.npy", new_train_dataY)
np.save("../data/angle_mnist.npy", new_train_dataAng)
