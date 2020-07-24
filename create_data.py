import argparse
import os
import tempfile

from preprocess.preprocess_extract_skeleton import extract_skeleton_from
from preprocess.preprocess_opjson import process_json_to_split

# define input params for parser
parser = argparse.ArgumentParser(
    description="")
parser.add_argument('--root_path', help='Path to the root of the data set. We assume that the data is sorted into classes')
parser.add_argument('--output_path', help='Path to the output directory.')
parser.add_argument('--video_formats', help='Formats to look for in directories. Everything else will be ignored. Split by |. (avi|mp4)')
parser.add_argument('--sample_size', help='Number of frames for one motion image', default=60)
parser.add_argument('--joint_number', help='Number of joints to expect', default=25)
parser.add_argument('--steps', help='Param to overlap data to augment. By default it should match sample_size', default=60)

args = parser.parse_args()
nchannels = 3


# debug to test the functionalities
debug = False
if debug:
    # fake user input
    args.root_path = os.path.abspath(r"./data/json")
    args.output_path = os.path.abspath(r"./data/trainNew")
    args.video_format = "avi"

# get classes
classifications = os.listdir(args.root_path)

# iterate over each class
for cl in classifications:
    # path to the output folder for the class
    output_path = os.path.join(args.output_path, cl)
    # path to the input folder of the class
    path_to_videos = os.path.join(args.root_path, cl)
    # create dir in output path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    videos = [(os.path.splitext(video)[0], os.path.join(path_to_videos, video)) for video in os.listdir(path_to_videos) if any([video.endswith(vf) for vf in args.video_format.split("|")])]
    # make temp dir to save data from OpenPose
    tmp_dir = tempfile.mkdtemp()
    for video_name, path_to_video in videos:
        extract_skeleton_from(path_to_video, os.path.join(tmp_dir, video_name))
    # iterate over the output from open pose and create motion images from data
    for video_folder in os.listdir(tmp_dir):
        process_json_to_split(os.path.join(tmp_dir, video_folder), output_path, video_folder, args.sample_size, args.joint_number, nchannels)


