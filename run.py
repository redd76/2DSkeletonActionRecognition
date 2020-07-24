import argparse
import glob
import os

import cv2
import numpy as np

import preprocess.preprocess_opjson as pre_json
import preprocess.preprocess_extract_skeleton as pre
import utils.model_utils as mutils
from datagen.DataGenerator import DataGenerator
import tempfile

# define params from parser
parser = argparse.ArgumentParser(
    description="")
parser.add_argument('--model_path', help='Path to the model.')
parser.add_argument('--weight_path', help='Path to the weights.')
parser.add_argument('--video_path', help='Path to the video.')
parser.add_argument('--model_type', help='Name of the trained model. AlexNet|DenseNet|ResNet|TwoStreamNet|SimpleCNNet|NNet')
parser.add_argument('--output_path', help='Path to write output video to.')

args = parser.parse_args()

# debug to test the functionalities
debug = True
if debug:
    # fake user input
    args.video_path = os.path.abspath(r"./data/test/test06.mp4")
    args.model_path = os.path.abspath(r"data/model_output/nnet_ep50_b25_dCustom_cl3/model.json")
    args.weight_path = os.path.abspath(r"data/model_output/nnet_ep50_b25_dCustom_cl3/weights.hd5")
    args.model_type = "DenseNet"
    args.output_path = r"./doc/images/test02_{}_cl3.avi".format(args.model_type)
    debug_output = r"./data/test/"
steps = 15

# classifications
classifications = ["stand",
                   "walk",
                   "sit",
                   "wave",
                   "punch"]

# create temp directories to store data
#json_tmp_dir = tempfile.mkdtemp()
#processed_tmp_dir = tempfile.mkdtemp()
#json_tmp_dir = r"C:\Users\Phillip\AppData\Local\Temp\tmps6t1d_lm"
#processed_tmp_dir = r"C:\Users\Phillip\AppData\Local\Temp\tmpz8m8sufc"
json_tmp_dir = r"C:\Users\Phillip\AppData\Local\Temp\tmpd5_xk_gl"
processed_tmp_dir = r"C:\Users\Phillip\AppData\Local\Temp\tmpx2n84izk"
# extract data with OpenPose and save to tmp dir
#pre.extract_skeleton_from(args.video_path, json_tmp_dir, '--video', write_images=processed_tmp_dir)

# load model from path
model = mutils.load_model_from_json(args.model_path, args.weight_path)
# load weights from path
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# verify input dimensions
#_, input_frames, input_joints, input_channel = model.layers[0].input_shape
input_frames, input_joints, input_channel = 60, 25, 3

video_as_opdata = []
# iterate over the json output of openpose for complete video
for jfile in os.listdir(json_tmp_dir):
    # get row data
    row = pre_json.get_image_data_from_json(os.path.join(json_tmp_dir, jfile), input_joints, input_channel)
    video_as_opdata.append(row)
video_as_opdata = np.asarray(video_as_opdata)
# split the complete image into the Input data format
split_images = pre_json.split_image_into_sample(img=video_as_opdata, sample_frames=60, n_joints=25, n_channels=3, steps=steps)

if debug:
    cv2.imwrite(debug_output + "complete.jpg", video_as_opdata)
    for i, img in enumerate(split_images):
        cv2.imwrite(debug_output + "img{0}.png".format(i), img)

# get the net input
input_images = []
for img in split_images:
    adjusted_img = DataGenerator.image_to_net_input(img, args.model_type, (input_joints, input_frames))
    input_images.append(adjusted_img)
# if Two inputs were found split them into sperate arrays for input
if len(input_images[0]) == 2:
    input1 = [img[0] for img in input_images]
    input2 = [img[1] for img in input_images]
    predictions = model.predict([np.asarray(input1), np.asarray(input2)])
else:
    predictions = model.predict(np.asarray(input_images))
# get the labels for every 60 frames
predicted_labels = [((i*steps, i*steps + 60), classifications[np.argmax(pred)]) for i, pred in enumerate(predictions)]
print(predicted_labels)
# write the labels onto processed images from openpose to output as new video
processed_images =[cv2.imread(p) for p in glob.glob(os.path.join(processed_tmp_dir, "*.png"))]
h, w, c = processed_images[0].shape
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter(args.output_path, fourcc, 30, (w, h))
for f, img in enumerate(processed_images):
    labels_for_frame = []
    for start_end, label in predicted_labels:
        if f >= start_end[0] and f <= start_end[1]:
            labels_for_frame.append(label)
    label_string = "|".join(labels_for_frame)
    img_wo_label = processed_images[f]
    img_w_label = cv2.putText(img_wo_label, label_string, (50, 50), cv2.FONT_HERSHEY_COMPLEX,
                              1, (255, 0, 0), 2, cv2.LINE_AA)
    out.write(img_w_label)

"""
#out = cv2.VideoWriter(args.output_path, 0, 1, (w,h))
for i, label in enumerate(predicted_labels):
    # Using cv2.putText() method
    for f in range(i*60, i*60+60):
        if f > len(processed_images)-1:
            break
        img_wo_label = processed_images[f]
        img_w_label = cv2.putText(img_wo_label, label, (50, 50), cv2.FONT_HERSHEY_COMPLEX,
                                  1, (255, 0, 0), 2, cv2.LINE_AA)
        out.write(img_w_label)

"""
cv2.destroyAllWindows()
out.release()









