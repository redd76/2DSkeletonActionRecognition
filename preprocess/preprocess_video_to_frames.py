import cv2
import os

root_path = r"C:/Users/Phillip/Pictures/customdata/"

clip_data = [(os.path.splitext(clip)[0], root_path + d + r"/", root_path + d + r"/" + clip) for d in os.listdir(root_path) for clip in os.listdir(root_path + d)]

for clip_name, clip_dir_path, clip_path in clip_data:
    vidcap = cv2.VideoCapture(clip_path)
    success = True
    count = 0
    while success:
        output_file = clip_dir_path + clip_name + r"/" + "frame%d.jpg" % count
        if os.path.exists(output_file):
            continue
        if not os.path.exists(clip_dir_path + clip_name):
            os.makedirs(clip_dir_path + clip_name)
        success, image = vidcap.read()
        cv2.imwrite(output_file, image)  # save frame as JPEG file
        count += 1