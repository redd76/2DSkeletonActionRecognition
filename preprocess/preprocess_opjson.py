import os
import subprocess
import json
import numpy as np
import cv2
import utils.normalization_utils as norm_utils

def get_image_data_from_json(file_name, n_joints=25, n_channels=3):
    with open(os.path.join(file_name), 'r', encoding='utf8') as f:
        data = json.load(f)
    people_in_scene = data["people"]
    if not len(people_in_scene) > 0:
        return None
    keypoints = people_in_scene[0]["pose_keypoints_2d"]
    # turn into image array
    keypoints_arr = np.asarray(keypoints).reshape((-1, 3))
    return keypoints_arr[0:n_joints, :]

def json_dir_to_image(directory):
    image_data = []
    json_files = [f for f in os.listdir(directory) if f.endswith(".json")]
    for filename in json_files:
        reshaped_arr = get_image_data_from_json(os.path.join(directory, filename))
        if reshaped_arr is None:continue
        # normalize base on bounding box
        image_data.append(reshaped_arr)
    if not len(image_data):
        return None
    return np.asarray(image_data)

def split_image_into_sample(img, sample_frames, n_joints, n_channels, steps=60):
    samples = []
    h, w, n_channel = img.shape
    for ID, i in enumerate(range(0, h, steps)):
        new_img = np.empty((sample_frames, n_joints, n_channels))
        if i + sample_frames > h:
            tmp_h, tmp_w, tmp_channels = img[i:i + sample_frames].shape
            # compute center offset
            yy = int((sample_frames - tmp_h) / 2)
            # copy img image into center of result image
            img_slice = img[i:h, :, :]
            new_img[yy:yy + tmp_h, :, :] = img_slice
        else:
            new_img[0:sample_frames, :, :] = img[i:i + sample_frames, :, :]
        samples.append(norm_utils.normalize_image_based_on_nose(new_img))
    return samples

def process_json_to_split(input_path, output_path, clip_name, sample_frames, n_joints, n_channels, steps=60):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    img = json_dir_to_image(input_path)
    if img is None:return
    samples = split_image_into_sample(img, sample_frames, n_joints, n_channels, steps)
    for ID, sample in enumerate(samples):
        cv2.imwrite(os.path.join(output_path, clip_name + "_{0}.jpg".format(str(ID))), sample)

if __name__ == '__main__':
    output_path = r"../data/trainTest/"
    json_path = r"../data/json/"
    nr_of_people = ["np1"]
    quality = ["goo"]
    clip_data = [(d, clip_dir, json_path + d + r"/" + clip_dir, os.path.join(output_path, d)) for d in os.listdir(json_path) for clip_dir in os.listdir(json_path + d) if any([nr_tok in clip_dir.split("_") for nr_tok in nr_of_people]) and any([quality_token in clip_dir.split("_") for quality_token in quality])]
    #print([x for x in clip_data if x[0] == "punch" and "custom" in x[1].split("_")])#
    sample_frames, n_joints, n_channels = 60, 25, 3
    img_count = 0
    for cl, clip_name, clip_path, output_path in clip_data:
        process_json_to_split(clip_path, output_path, clip_name, sample_frames, n_joints, n_channels, 15)

    print(img_count)

