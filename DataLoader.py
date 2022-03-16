import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import os
import random


class DataLoader:
    def __init__(self, image_shape, batch_size):
        self.image_size = image_shape[:2]
        self.batch_size = batch_size
        self.ROOT = '/media/bonilla/HDD_2TB_basura/databases/FacesKeypoints/'
        self.ids = sorted(glob.glob(os.path.join(self.ROOT, '*')))

    def _load_image(self, img_path, mask_path):
        random_numbers = np.random.rand(2)
        image = cv2.imread(img_path)
        image = cv2.resize(image, self.image_size)
        if random_numbers[0] < 0.5:
            image = cv2.flip(image, 1)
        if random_numbers[1] < 0.5:
            R = cv2.getRotationMatrix2D((self.image_size[0] // 2, self.image_size[1] // 2), np.random.randint(-45, 45), 1)
            image = cv2.flip(image, 1)
            image = cv2.warpAffine(image, R, self.image_size, borderMode=cv2.BORDER_REPLICATE)
        image = (image.astype(np.float32) - 127.5) / 127.5
        if mask_path is not None:
            mask = cv2.imread(mask_path)
            mask = cv2.resize(mask, self.image_size)
            if random_numbers[0] < 0.5:
                mask = cv2.flip(mask, 1)
            if random_numbers[1] < 0.5:
                mask = cv2.flip(mask, 1)
                mask = cv2.warpAffine(mask, R, self.image_size)
            mask = mask[:, :, 0].astype(np.float32) / 255.
            mask = np.expand_dims(mask, axis=-1)
            return image, mask
        return image

    def load_batch(self, batch_size=None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        rand_idxs = random.sample(range(len(self.ids)), batch_size)

        images_out = []
        masks_input = []
        images_input = []
        for idx in rand_idxs:
            selected_id = self.ids[idx]
            id_images = sorted(glob.glob(os.path.join(self.ROOT, selected_id, 'images', '*')))
            id_keypoints = sorted(glob.glob(os.path.join(self.ROOT, selected_id, 'keypoints', '*')))
            rand_photos = random.sample(range(len(id_images)), 2)
            i1, m1 = self._load_image(id_images[rand_photos[0]], id_keypoints[rand_photos[0]])
            i2 = self._load_image(id_images[rand_photos[1]], None)
            images_out.append(i1)
            masks_input.append(m1)
            images_input.append(i2)

        return np.array(images_input), np.array(masks_input), np.array(images_out)


if __name__ == '__main__':
    d = DataLoader((128, 128, 3), 16)
    imgs_in, masks_in, imgs_out = d.load_batch()
    print()
