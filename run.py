import utils.model_utils as mutils
import argparse
import preprocess.preprocess_extract_skeleton as pre
import preprocess.preprocess_opjson as pre_json
from datagen.DataGenerator import DataGenerator

import tempfile
import os
import numpy as np
import cv2
import shutil
import glob

parser = argparse.ArgumentParser(
    description="")
parser.add_argument('--model_path', help='Path to the model.')
parser.add_argument('--weight_path', help='Path to the weights.')
parser.add_argument('--video_path', help='Path to the video.')
parser.add_argument('--model_type', help='Name of the trained model. AlexNet|DenseNet|TwoStreamNet|SimpleCNNet|NNet')
parser.add_argument('--output_path', help='Path to write output video to.')

args = parser.parse_args()


test = False
if test:
    # fake user input
    args.video_path = os.path.abspath(r"./data/test/test.avi")
    args.model_path = os.path.abspath(r"./data/model_output/20200719190658/model.json")
    args.weight_path = os.path.abspath(r"./data/model_output/20200719190658/weights.hd5")
    args.model_type = "TwoStreamNet"
    args.output_path = r"C:\Users\Phillip\Pictures\BoxingTipsHowtoBuildPunchingPower_punch_u_cm_np1_fr_goo_2.avi"


classifications = ["stand",
                   "walk",
                   "squat",
                   "wave",
                   "punch"]

json_tmp_dir = tempfile.mkdtemp()
processed_tmp_dir = tempfile.mkdtemp()
pre.extract_skeleton_from(args.video_path, json_tmp_dir, '--video', write_images=processed_tmp_dir)

model = mutils.load_model_from_json(args.model_path, args.weight_path)
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
_, input_frames, input_joints, input_channel = model.layers[0].input_shape

video_as_opdata = []

for jfile in os.listdir(json_tmp_dir):
    row = pre_json.get_image_data_from_json(os.path.join(json_tmp_dir, jfile), input_joints, input_channel)
    video_as_opdata.append(row)

video_as_opdata = np.asarray(video_as_opdata)
split_images = pre_json.split_image_into_sample(img=video_as_opdata, sample_frames=60, n_joints=input_joints,n_channels=input_channel)


input_images = []
for img in split_images:
    adjusted_img = DataGenerator.image_to_net_input(img, args.model_type, (input_frames, input_joints))
    input_images.append(adjusted_img)

predictions = model.predict(input_images)
predicted_labels = [classifications[np.argmax(pred)] for pred in predictions]


processed_images =[cv2.imread(p) for p in glob.glob(os.path.join(processed_tmp_dir, "*.png"))]

h, w, c = processed_images[0].shape
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter(args.output_path, fourcc, 30, (w, h))
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

cv2.destroyAllWindows()
out.release()









