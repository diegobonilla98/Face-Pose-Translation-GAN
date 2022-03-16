import os
import glob
import uuid

import dlib
import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
from urllib.request import urlopen, Request


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

ROOT = '/media/bonilla/HDD_2TB_basura/databases/VGG_face/vgg_face_dataset/files'
all_files = glob.glob(os.path.join(ROOT, '*.txt'))
with tqdm.tqdm(total=len(all_files)) as pbar:
    for idx, file_path in enumerate(all_files[1546:]):
        idx += 1546
        with open(file_path, 'r') as file:
            data = file.read().split('\n')
        valid_images = 0
        os.mkdir(os.path.join('/media/bonilla/HDD_2TB_basura/databases/FacesKeypoints', f'id_{idx}'))
        os.mkdir(os.path.join('/media/bonilla/HDD_2TB_basura/databases/FacesKeypoints', f'id_{idx}', 'images'))
        os.mkdir(os.path.join('/media/bonilla/HDD_2TB_basura/databases/FacesKeypoints', f'id_{idx}', 'keypoints'))
        for entity in data:
            if valid_images >= 15:
                break
            attrs = entity.split(' ')
            url = attrs[1]

            try:
                req = Request(url, headers={'User-Agent': 'Chrome/51.0.2704.103'})
                resp = urlopen(req, timeout=1)
                image = np.asarray(bytearray(resp.read()), dtype="uint8")
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                faces = detector(gray, 1)[0]
                keypoints = predictor(gray, faces)
                keypoints = [(keypoints.part(idx).x, keypoints.part(idx).y) for idx in range(68)]

                min_x = np.clip(min(keypoints, key=lambda x: x[0])[0], 0, image.shape[1])
                min_y = np.clip(min(keypoints, key=lambda x: x[1])[1], 0, image.shape[0])
                max_x = np.clip(max(keypoints, key=lambda x: x[0])[0], 0, image.shape[1])
                max_y = np.clip(max(keypoints, key=lambda x: x[1])[1], 0, image.shape[0])
                c = ((max_x - min_x) + (max_y - min_y)) / 2
                size = int(c / 60)

                mask = np.zeros_like(image)

                for (x, y) in keypoints:
                    cv2.circle(mask, (x, y), size, (255, 255, 255), -1)

                mask = mask[min_y: max_y, min_x: max_x]
                image = image[min_y: max_y, min_x: max_x]

                assert 0 not in image.shape
                assert 0 not in mask.shape
            except:
                continue

            name = uuid.uuid4()
            cv2.imwrite(os.path.join('/media/bonilla/HDD_2TB_basura/databases/FacesKeypoints', f'id_{idx}', 'images', f'{name}.jpg'), image)
            cv2.imwrite(os.path.join('/media/bonilla/HDD_2TB_basura/databases/FacesKeypoints', f'id_{idx}', 'keypoints', f'{name}.jpg'), mask)

            valid_images += 1

        pbar.update()


