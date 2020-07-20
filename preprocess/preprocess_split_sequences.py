import os
import subprocess
import json
import numpy as np
import cv2

root_path = r"C:/Users/Phillip/Documents/beuth/customdata/"
frames_path = root_path + r"frames/"
clip_squence_img_path = r"C:/Users/Phillip/Documents/beuth/customdata/outputNew"
clip_data = [(os.path.splitext(img)[0], clip_squence_img_path + r"/" + img, root_path + r"/xdata3/") for img in os.listdir(clip_squence_img_path)]
sample_frames = 150
for image_name, image_path, output_path in clip_data:
    classification = image_name.split("_")[0]
    if not os.path.exists(output_path + classification):
        os.makedirs(output_path + classification)

    img = cv2.imread(image_path)
    h, w, n_channel = img.shape
    new_img = X = np.empty((sample_frames, 25, 3))
    for ID, i in enumerate(range(0, h, 30)):
        if i+sample_frames > h:
            break
        new_img[0:150] = img[i:i+sample_frames]
        cv2.imwrite(os.path.join(output_path, classification, classification+"_{0}.jpg".format(str(ID))), new_img)

