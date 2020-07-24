import os
import subprocess

openpose_location = r"C:\Users\Phillip\Documents\beuth\2DSkeletonActionClassification\openpose"

def extract_skeleton_from(input_path, output_path, input_type='--video', write_images=None):
    os.chdir(openpose_location)
    render_pose_value = 0 if not write_images else 2
    popen_args = ['../openpose/bin/OpenPoseDemo.exe', input_type, input_path, '--write_json', output_path, '--display', '0',
         '--render_pose', str(render_pose_value), '--number_people_max', '1']
    if write_images is not None:
        popen_args.append('--write_images')
        popen_args.append(write_images)
    process = subprocess.call(popen_args)

if __name__ == '__main__':
    root_path = r"../data/source/hd60"
    frames_path = root_path + r"/"
    output_path = r"../custom/hd60_json/"

    op_flag = "--video"

    clip_data = [(clip_dir, frames_path + d + r"/" + clip_dir, output_path + d + r"/" + clip_dir) for d in os.listdir(frames_path) for clip_dir in os.listdir(frames_path + d)]
    for clip_name, clip_dir_path, output_path in clip_data:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        extract_skeleton_from(clip_dir_path, output_path, op_flag)