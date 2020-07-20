import os
import subprocess
import json
import numpy as np
import cv2

def get_image_data_from_json(file_name, n_joints=25, n_channels=3):
    data = None
    with open(os.path.join(file_name), 'r', encoding='utf8') as f:
        data = json.load(f)
    people_in_scene = data["people"]
    if not len(people_in_scene) > 0:
        black = np.empty((n_joints, n_channels))
        return  black
    keypoints = people_in_scene[0]["pose_keypoints_2d"]
    # turn into image array
    keypoints_arr = np.asarray(keypoints).reshape((-1, 3))
    return keypoints_arr[0:n_joints, :]

def json_dir_to_image(directory):
    image_data = []
    json_files = [f for f in os.listdir(directory) if f.endswith(".json")]
    for filename in json_files:
        reshaped_arr = get_image_data_from_json(os.path.join(directory, filename))
        # normalize base on bounding box
        normalized = normalize_frame(reshaped_arr)
        image_data.append(normalized)
    if not len(image_data):
        return None
    return np.asarray(image_data)

def split_image_into_sample(img, sample_frames, n_joints, n_channels):
    samples = []
    print(img.shape)
    h, w, n_channel = img.shape
    new_img = np.empty((sample_frames, n_joints, n_channels))
    for ID, i in enumerate(range(0, h, sample_frames)):
        if i + sample_frames > h:
            tmp_h, tmp_w, tmp_channels = img[i:i + sample_frames].shape
            color = (0.0, 0.0, 0.0)
            padded = np.full((sample_frames, n_joints, n_channels), color, dtype=np.uint8)
            # compute center offset
            yy = (sample_frames - tmp_h) // 2
            # copy img image into center of result image
            padded[yy:yy + tmp_h, 0:n_joints] = img[i:i + sample_frames]
            new_img[0:sample_frames] = padded
        else:
            new_img[0:sample_frames] = img[i:i + sample_frames]
        samples.append(new_img)
    return samples

def remap(value, maxInput, minInput, maxOutput, minOutput):
    return ( (value - minInput) / (maxInput - minInput) ) * (maxOutput - minOutput) + minOutput

def normalize_frame(frame):
    # filter zero values
    non_zero = frame[np.any(frame != [0, 0, 0], axis=-1)]
    x_min = np.amin(non_zero[:, 0])
    y_min = np.amin(non_zero[:, 1])

    x_max = np.amax(non_zero[:, 0])
    y_max = np.amax(non_zero[:, 1])
    len_x = x_max - x_min
    len_y = y_max - y_min

    bounding_box_center = (x_max - (len_x / 2.0), y_max - (len_y / 2.0))
    bounding_box_max = ((len_x/2.0), (len_y/2.0))
    bounding_box_min = (-(len_x/2.0), -(len_y/2.0))
    # put all joint positions relative to bounding box
    nframe = []
    for jpositions in frame:
        jpositions[0] -= bounding_box_center[0]
        jpositions[1] -= bounding_box_center[1]
        # remap to rgb
        njposition = [0,0,0]

        njposition[0] = remap(jpositions[0], bounding_box_max[0], bounding_box_min[0], 0.0, 255.0)
        njposition[1] = remap(jpositions[1], bounding_box_max[1], bounding_box_min[1], 0.0, 255.0)
        njposition[2] = remap(jpositions[2], 1.0, 0.0, 0.0, 255.0)

        nframe.append(njposition)
    return np.asarray(nframe)

def process_json_to_split(classification, clip, input_path, output_path, sample_frames, n_joints, n_channels):
    output_path = os.path.join(output_path, classification)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    img = json_dir_to_image(input_path)
    if img is None:return
    samples = split_image_into_sample(img, sample_frames, n_joints, n_channels)
    for ID, sample in enumerate(samples):
        print("Writing to {0}".format(output_path))
        cv2.imwrite(os.path.join(output_path, classification + clip + "_{0}.jpg".format(str(ID))), sample)
"""
if __name__ == '__main__':
    output_path = r"../data/train2/"
    json_path = r"../data/json/"
    nr_of_people = ["np1"]
    quality = ["med", "goo"]
    clip_data = [(d, clip_dir, json_path + d + r"/" + clip_dir, output_path) for d in os.listdir(json_path) for clip_dir in os.listdir(json_path + d) if any([nr_tok in clip_dir.split("_") for nr_tok in nr_of_people]) and any([quality_token in clip_dir.split("_") for quality_token in quality])]

    sample_frames, n_joints, n_channels = 60, 25, 3
    img_count = 0
    for classification, clip, clip_dir_path, output_path in clip_data:
        process_json_to_split(clip_dir_path, output_path, sample_frames, n_joints, n_channels)

    print(img_count)
"""

if __name__ == '__main__':
    output_path = r"../data/train2/"
    json_path = r"C:\Users\Phillip\Documents\beuth\2DSkeletonActionClassification\data\json\punch\punch_actor2_take01"
    clip_name = r'_custom_punch_np1_fr_goo_a2_take01'
    #nr_of_people = ["np1"]
    #quality = ["med", "goo"]
    #clip_data = [(d, clip_dir, json_path + d + r"/" + clip_dir, output_path) for d in os.listdir(json_path) for clip_dir in os.listdir(json_path + d) if any([nr_tok in clip_dir.split("_") for nr_tok in nr_of_people]) and any([quality_token in clip_dir.split("_") for quality_token in quality])]

    sample_frames, n_joints, n_channels = 60, 25, 3
    img_count = 0
    process_json_to_split('punch', clip_name, json_path, output_path, sample_frames, n_joints, n_channels)

    print(img_count)

