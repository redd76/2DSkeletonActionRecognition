# To use Inference Engine backend, specify location of plugins:
# export LD_LIBRARY_PATH=/opt/intel/deeplearning_deploymenttoolkit/deployment_tools/external/mklml_lnx/lib:$LD_LIBRARY_PATH
import cv2 as cv
import numpy as np
import argparse
# import imutils
import time
import os


def getInputImage(cap, BODY_PARTS, max_frames=150):
    quant_img = []
    frameWidth = None
    frameHeight = None
    frame_number = 0
    while (cap.isOpened()) and frame_number < max_frames:
        ret, frame = cap.read()
        if frame is None:
            break
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        cmin, cmaxX, cmaxY = 0, frameWidth, frameHeight

        inp = cv.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                   (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inp)
        start_t = time.time()
        out = net.forward()

        #print("time is ", time.time() - start_t)
        # print(inp.shape)
        # kwinName = "Pose Estimation Demo: Cv-Tricks.com"
        # cv.namedWindow(kwinName, cv.WINDOW_AUTOSIZE)
        # assert(len(BODY_PARTS) == out.shape[1])
        points = []
        quant = []

        for i in range(len(BODY_PARTS)):
            # Slice heatmap of corresponging body's part.
            heatMap = out[0, i, :, :]

            # Originally, we try to find all the local maximums. To simplify a sample
            # we just find a global one. However only a single pose at the same time
            # could be detected this way.
            _, conf, _, point = cv.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]
            reframeX = 255 / cmaxX * x
            reframeY = 255 / cmaxY * y
            # Add a point if it's confidence is higher than threshold.
            points.append((int(x), int(y), conf))
            quant.append((reframeX, reframeY, conf))
        quant_img.append(quant)
        frame_number += 1
    quant_arr = np.asarray(quant_img)
    wd, ht, cc = quant_arr.shape
    ww = 150
    hh = len(BODY_PARTS)
    #print(ht, wd, ww, hh)
    color = (0.0, 0.0, 0.0)
    result = np.full((ww, hh, cc), color, dtype=np.uint8)
    #print(result.shape, quant_arr.shape)
    # compute center offset
    xx = (ww - wd) // 2
    yy = (hh - ht) // 2
    # copy img image into center of result image
    #print(xx + wd, yy + ht)
    result[xx:xx+wd, 0:hh] = quant_arr
    return result

parser = argparse.ArgumentParser(
    description='This script is used to demonstrate OpenPose human pose estimation network '
                'from https://github.com/CMU-Perceptual-Computing-Lab/openpose project using OpenCV. '
                'The sample and model are simplified and could be used for a single person on the frame.')
parser.add_argument('--input', help='Path to input image.')
parser.add_argument('--proto', help='Path to .prototxt')
parser.add_argument('--model', help='Path to .caffemodel')
parser.add_argument('--dataset', help='Specify what kind of model was trained. '
                                      'It could be (COCO, MPI) depends on dataset.')
parser.add_argument('--thr', default=0.1, type=float, help='Threshold value for pose parts heat map')
parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')

args = parser.parse_args()

args.model = r"C:\Users\Phillip\Documents\beuth\openpose\models\pose\body_25\pose_iter_584000.caffemodel"
args.proto =r"C:\Users\Phillip\Documents\beuth\openpose\models\pose\body_25\pose_deploy.prototxt"

if args.dataset == 'COCO':
    BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                  "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                  "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                  "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

    POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                  ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                  ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                  ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                  ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]
elif args.dataset == 'MPI':
    # assert(args.dataset == 'MPI')
    BODY_PARTS = {"Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                  "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                  "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                  "Background": 15}

    POSE_PAIRS = [["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                  ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                  ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                  ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"]]
else:

    BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4, "LShoulder": 5, "LElbow": 6,
                  "LWrist": 7, "MidHip": 8, "RHip": 9, "RKnee": 10, "RAnkle": 11, "LHip": 12, "LKnee": 13, "LAnkle": 14,
                  "REye": 15, "LEye": 16, "REar": 17, "LEar": 18, "LBigToe": 19, "LSmallToe": 20, "LHeel": 21,
                  "RBigToe": 22, "RSmallToe": 23, "RHeel": 24, "Background": 25}

    POSE_PAIRS = [["Neck", "MidHip"], ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                  ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"], ["MidHip", "RHip"],
                  ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["MidHip", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"],
                  ["Neck", "Nose"], ["Nose", "REye"], ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"],
                  ["RShoulder", "REar"], ["LShoulder", "LEar"], ["LAnkle", "LBigToe"], ["LBigToe", "LSmallToe"],
                  ["LAnkle", "LHeel"], ["RAnkle", "RBigToe"], ["RBigToe", "RSmallToe"], ["RAnkle", "RHeel"]]

inWidth = args.width
inHeight = args.height

net = cv.dnn.readNetFromCaffe(args.proto, args.model)

# assuming training data is comprised of avi
classifications = {"stand":[1.0, 0.0, 0.0, 0.0, 0.0],
                   "sit":[0.0, 1.0, 0.0, 0.0, 0.0],
                   "walk":[0.0, 0.0, 1.0, 0.0, 0.0],
                   "run": [0.0, 0.0, 0.0, 1.0, 0.0],
                  "jump": [0.0, 0.0, 0.0, 0.0, 1.0]}
source_data = r"D:/dataset2/train/datax/source/walk/"
preprocessed_output = r"D:/dataset2/train/datax/preprocessed4/"
#imgs = [(dir, source_data + dir + r"/" + avi, os.path.splitext(avi)[0]) for dir in os.listdir(source_data) for avi in os.listdir(source_data + dir)]
imgs = [("walk", source_data + avi, os.path.splitext(avi)[0]) for avi in os.listdir(source_data)]

nr_of_people = ["np1"]
quality = ["med", "goo", "bad"]
# exclude low quality and more than 1 actor
imgs = [(dir, path, name) for dir, path, name in imgs if any([nr_tok in name.split("_") for nr_tok in nr_of_people]) and any([quality_token in name.split("_") for quality_token in quality])]

for dir, source_path, file_name in imgs:
    starttime = time.time()
    output_file = preprocessed_output + dir + r"/" + file_name + ".jpg"
    if os.path.exists(output_file):
        continue
    if not os.path.exists(preprocessed_output + dir):
        os.makedirs(preprocessed_output + dir)
    cap = cv.VideoCapture(source_path)
    img = getInputImage(cap, BODY_PARTS, max_frames=150)
    cv.imwrite(output_file, np.asarray(img))
    print(time.time() - starttime)