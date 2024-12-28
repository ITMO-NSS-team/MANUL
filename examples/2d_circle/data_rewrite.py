import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def circle_curve():
    im = cv2.imread("data/circle_curve.jpg")
    im = np.sum(im/255, axis=2)[10:-10, 25:-25]

    xs = np.arange(0, im.shape[0], 2)
    ys = np.arange(0, im.shape[1], 2)
    grid = np.meshgrid(xs, ys)
    grid_colors = im[grid[0], grid[1]]

    grid[0] = grid[0][grid_colors < 2.3]/np.max(grid[0])
    grid[1] = grid[1][grid_colors < 2.3]/np.max(grid[1])
    grid_colors = grid_colors[grid_colors < 2.3]/np.max(grid_colors)

    #plt.imshow(im)
    plt.scatter(grid[1], grid[0], c=grid_colors, s=0.5, cmap='Reds')
    plt.show()


    df = pd.DataFrame()
    df['x'] = np.ravel(grid[0])
    df['y'] = np.ravel(grid[1])
    df['color'] = np.ravel(grid_colors)

    test_inds = np.random.choice(range(len(df)), int(len(df)*0.2), replace=False)
    test = df.loc[test_inds]

    train_inds = np.delete(np.arange(len(df)), test_inds)
    train = df.loc[train_inds]

    plt.scatter(test['y'], test['x'], c=test['color'], s=0.5, cmap='Reds')
    plt.show()

    plt.scatter(train['y'], train['x'], c=train['color'], s=0.5, cmap='Reds')
    plt.show()

    train.to_csv('data/circle_curve_train.csv', index=False)
    test.to_csv('data/circle_curve_test.csv', index=False)

def circle_straight():
    im = cv2.imread("data/circle_straight.png")
    im = np.sum(im/255, axis=2)[:, 25:-25].astype(int)

    xs = np.arange(0, im.shape[0], 10)
    ys = np.arange(0, im.shape[1], 10)
    grid = np.meshgrid(xs, ys)
    grid_colors = im[grid[0], grid[1]]

    grid[0] = grid[0][grid_colors < 2.3]/np.max(grid[0])
    grid[1] = grid[1][grid_colors < 2.3]/np.max(grid[1])
    grid_colors = grid_colors[grid_colors < 2.3]/np.max(grid_colors)

    #plt.imshow(im)
    plt.scatter(grid[1], grid[0], c=grid_colors, s=0.5, cmap='Reds')
    plt.show()


    df = pd.DataFrame()
    df['x'] = np.ravel(grid[0])
    df['y'] = np.ravel(grid[1])
    df['color'] = np.ravel(grid_colors)

    test_inds = np.random.choice(range(len(df)), int(len(df)*0.2), replace=False)
    test = df.loc[test_inds]

    train_inds = np.delete(np.arange(len(df)), test_inds)
    train = df.loc[train_inds]

    plt.scatter(test['y'], test['x'], c=test['color'], s=0.5, cmap='Reds')
    plt.show()

    plt.scatter(train['y'], train['x'], c=train['color'], s=0.5, cmap='Reds')
    plt.show()

    train.to_csv('data/circle_straight_train.csv', index=False)
    test.to_csv('data/circle_straight_test.csv', index=False)

circle_straight()