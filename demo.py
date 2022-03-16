from tensorflow.keras.models import load_model
import cv2
import numpy as np
import dlib
from FacePoseGAN import FacePoseGAN
import tensorflow.keras.backend as K
import tensorflow as tf


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)


gan = FacePoseGAN(is_demo=True)
generator = gan.generator
weights_path = './RESULTS/weights/epoch_9900.h5'
generator.load_weights(weights_path)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cam = cv2.VideoCapture(1)
w, h = cam.get(cv2.CAP_PROP_FRAME_WIDTH), cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

reference = cv2.imread('./faces/obama.jpg')
gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
faces = detector(gray, 1)[0]
keypoints = predictor(gray, faces)
keypoints = [(keypoints.part(idx).x, keypoints.part(idx).y) for idx in range(68)]
min_x = int(np.clip(min(keypoints, key=lambda x: x[0])[0], 0, w))
min_y = int(np.clip(min(keypoints, key=lambda x: x[1])[1], 0, h))
max_x = int(np.clip(max(keypoints, key=lambda x: x[0])[0], 0, w))
max_y = int(np.clip(max(keypoints, key=lambda x: x[1])[1], 0, h))
ref_face = reference[min_y: max_y, min_x: max_x]
ref_face = cv2.resize(ref_face, (128, 128))
ref_face = (ref_face.astype('float32') - 127.5) / 127.5

while True:
    ret, frame = cam.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    if len(faces) < 1:
        continue
    faces = faces[0]
    keypoints = predictor(gray, faces)
    keypoints = [(keypoints.part(idx).x, keypoints.part(idx).y) for idx in range(68)]
    canvas = frame.copy()
    mask = np.zeros_like(frame)

    min_x = int(np.clip(min(keypoints, key=lambda x: x[0])[0], 0, w))
    min_y = int(np.clip(min(keypoints, key=lambda x: x[1])[1], 0, h))
    max_x = int(np.clip(max(keypoints, key=lambda x: x[0])[0], 0, w))
    max_y = int(np.clip(max(keypoints, key=lambda x: x[1])[1], 0, h))
    c = ((max_x - min_x) + (max_y - min_y)) / 2
    size = int(c / 60)

    for x, y in keypoints:
        cv2.circle(canvas, (x, y), size, (0, 0, 255), -1)
        cv2.circle(mask, (x, y), size, (255, 255, 255), -1)

    my_mask = mask[min_y: max_y, min_x: max_x]
    my_mask = cv2.resize(my_mask, (128, 128))[:, :, 0] / 255.
    my_mask = np.expand_dims(my_mask, axis=-1)
    input_img = np.expand_dims(np.concatenate([ref_face, my_mask], axis=-1), axis=0)
    output = generator.predict(input_img)[0]

    output = np.uint8(((output + 1) * 0.5) * 255.)
    output = cv2.pyrUp(output)
    output = cv2.pyrUp(output)

    cv2.imshow('Webcam', canvas)
    cv2.imshow('Result', output)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
cam.release()
