import glob
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np


root = '/media/bonilla/HDD_2TB_basura/databases/FacesKeypoints/'
ids = sorted(glob.glob(os.path.join(root, '*')), key=lambda x: int(x.split(os.sep)[-1].split('_')[-1]))[71:]
for id_image in ids:
    images = glob.glob(os.path.join(id_image, 'images', '*.jpg'))
    images_arr = []
    for img in images:
        img = cv2.imread(img)
        img = cv2.resize(img, (256, 256))
        images_arr.append(img)

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()

    plt.title(id_image.split(os.sep)[-1])
    plt.imshow(np.hstack(images_arr)[:, :, ::-1])
    plt.axis('off')
    plt.show()
