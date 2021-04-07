import numpy as np
import cv2


def draw_bbox(image, bboxes, class_names):
    size = 2
    thickness = 4
    font = cv2.FONT_HERSHEY_SIMPLEX
    for index, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, bbox[:4])
        if x1 >= x2 or y1 >= y2:
            continue
        conf = bbox[-1]
        # print(x1, y1, x2, y2, image.shape, image.dtype)
        image = cv2.rectangle(image, (int(x1), int(y1)),
                              (int(x2), int(y2)), (255, 0, 0), thickness=thickness)
        image = cv2.putText(image, "{}:{:.3f}".format(class_names[index], conf), (int(x1), int(y1)+80),
                            font, size, (0, 0, 255), thickness=thickness)
    return image


def extract_bboxtlbr_from_mask(mask, color):
    pos_mask = mask[:, :, 0] > 200
    y, x = np.where(pos_mask)
    extreme_points = get_extreme_points(x, y)
    left, top, right, down = extreme_points

    x1 = left[0]
    y1 = top[1]
    x2 = right[0]
    y2 = down[1]

    return x1, y1, x2, y2


def get_extreme_coords(array1, array2, value_op, argvalue_op):
    max_v = value_op(array1)
    max_idx = argvalue_op(array1)
    other_v = array2[max_idx]

    return max_v, other_v


def get_extreme_points(x_c, y_c):

    left_x, left_y = get_extreme_coords(x_c, y_c, np.min, np.argmin)
    left = np.stack((left_x, left_y))
    top_y, top_x = get_extreme_coords(y_c, x_c, np.min, np.argmin)
    top = np.stack((top_x, top_y))
    right_x, right_y = get_extreme_coords(x_c, y_c, np.max, np.argmax)
    right = np.stack((right_x, right_y))
    down_y, down_x = get_extreme_coords(y_c, x_c, np.max, np.argmax)
    down = np.stack((down_x, down_y))
    extreme_points = np.stack((left, top, right, down))

    return extreme_points
