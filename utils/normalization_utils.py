import numpy as np

def remap(value, maxInput, minInput, maxOutput, minOutput):
    return ((value - minInput) / (maxInput - minInput) ) * (maxOutput - minOutput) + minOutput

def normalize_image(img):
    #non_zero = img[np.any(img != [0, 0, 0], axis=None)]
    # put all joint positions relative to bounding box
    nimg = []
    for f in range(img.shape[0]):
        frame = img[f,:,:]
        non_zero = frame[np.any(frame != [0, 0, 0], axis=-1)]
        non_zero = non_zero[np.all(np.isfinite(non_zero), axis=-1)]
        if non_zero.shape[0] == 0:
            nimg.append(np.empty(frame.shape))
            continue
        x_min = np.amin(non_zero[:, 0])
        y_min = np.amin(non_zero[:, 1])
        x_max = np.amax(non_zero[:, 0])
        y_max = np.amax(non_zero[:, 1])

        scale_coord_x = x_max - x_min
        scale_coord_y = y_max - y_min
        nframe = []
        for jpositions in img[f, :, :]:
            if all(jpositions == 0):
                nframe.append([0,0,0])
                continue
            # remap to rgb
            norm_x = ((jpositions[0] - x_min) / scale_coord_x) * 255
            norm_y = ((jpositions[1] - y_min) / scale_coord_y) * 255
            norm_c = jpositions[2] * 255.0
            nframe.append([norm_x, norm_y, norm_c])
        nimg.append(nframe)
    return np.asarray(nimg)


def normalize_image_based_on_nose(img):
    non_zero_indices = [i for i in range(img.shape[0]) if np.all(img[i, 0, :] != [0,0,0]) and np.all(np.isfinite(img[i, 0, :]))]
    non_zero = img[non_zero_indices, :, :]
    if non_zero.shape[0] == 0:
        return img
    # put all joint positions relative to bounding box
    nimg = np.empty(img.shape)
    for i in range(img.shape[0]):
        nframe = []
        frame = img[i, :, :]
        non_zero = frame[np.any(frame != [0, 0, 0], axis=-1)]
        non_zero = non_zero[np.all(np.isfinite(non_zero), axis=-1)]
        if non_zero.shape[0] == 0:
            nimg[i,:] = np.empty(frame.shape)
            continue
        ref_x = non_zero[0,0]
        ref_y = non_zero[0,1]
        max_x = np.amax(non_zero[:,0]) - ref_x
        min_x = np.amin(non_zero[:,0]) - ref_x
        max_y = np.amax(non_zero[:, 1]) - ref_y
        min_y = np.amin(non_zero[:, 1]) - ref_y
        scale_coord_x = max_x - min_x
        scale_coord_y = max_y - min_y
        for j in range(1, img.shape[1]):
            point = np.empty(3)
            point[0:2] = img[i, j, 0:2] - np.asarray([ref_x, ref_y])
            point[2] = img[i, j, 2]
            # remap to rgb
            norm_x = remap(point[0], max_x, min_x, 255.0, 0.0)
            norm_y = remap(point[1], max_y, min_y, 255.0, 0.0)
            norm_c = point[2] * 255
            # norm_x = norm_x
            # norm_y = norm_y
            # norm_c = norm_c
            nimg[i,j,:] = np.asarray([norm_x, norm_y, norm_c])
    return nimg

def normalize_frame(frame):
    # filter zero values
    is_all_zero = np.all((frame == [0, 0, 0]))
    if is_all_zero:
        return frame
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

        njposition[0] = remap(jpositions[0], bounding_box_max[0], bounding_box_min[0], 255.0, 0.0)
        njposition[1] = remap(jpositions[1], bounding_box_max[1], bounding_box_min[1], 255.0, 0.0)
        njposition[2] = remap(jpositions[2], 1.0, 0.0, 255.0, 0.0)

        nframe.append(njposition)
    return np.asarray(nframe)
