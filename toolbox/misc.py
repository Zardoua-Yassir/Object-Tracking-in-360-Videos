import time

from torch.nn.functional import conv2d
import numpy as np
import os
import cv2


def _compute_regression_targets(gt_bbox, x_grid, y_grid):
    """
    Computes the regression labels localizing the object within the srch_window region
    :param gt_bbox: top-left (x0, y0) and bottom-right (x1, y1) corner coordinates of the object given within the
    srch_window region
    :param x_grid:
    :param y_grid:
    :return: regression labels loc_tar
    """
    x0, y0, x1, y1 = gt_bbox

    left_target = x_grid - x0
    top_target = y_grid - y0
    right_target = x1 - x_grid
    bottom_target = y1 - y_grid

    reg_targets = np.stack((left_target, top_target, right_target, bottom_target),
                           axis=0)  # loc_tar shape = (4,
    # cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.OUTPUT_SIZE).
    return reg_targets


def compute_xy_grid(out_size, search_size, batch_size=None):
    """
    Computes two grids x_grid, y_grid, of size (out_height, out_width) in numpy using meshgrid.

    :param out_size: width and height of the model's output maps. Must be speficied by a single integer or a tuple
    of integers (out_height, out_width) in case the width is not equal to the height
    :param search_size: width and height of the input srch_window region (denoted in the original paper as X). Must be
    speficied by a single integer or a tuple of integers (search_height, search_width) in case the width is not
    equal to the height
    :param batch_size: if not None, x_grid and y_grid will have an dimension of size 'batch_size' at dim=0
    :return:
    x_grid: A 2D numpy array containing all x coordinates of the srch_window region, starting from 0 to search_width - 1
    , that correspond to i coordinates of the output map.
    y_grid: A 2D numpy array containing all y coordinates of the srch_window region, starting from 0 to search_height -
    1, that correspond to j coordinates of the output map.
    """
    out_height, out_width = out_size if isinstance(out_size, tuple) else (out_size, out_size)
    search_height, search_width = search_size if isinstance(search_size, tuple) else (search_size, search_size)

    x_range = np.linspace(0, search_width - 1, out_width)  # a vector of out_width elements, going from 0 to
    # search_width - 1 with uniform steps.
    y_range = np.linspace(0, search_height - 1, out_height)
    x_grid, y_grid = np.meshgrid(x_range, y_range)

    if batch_size is not None:
        # duplicate the grids on the mini-batch dimension (for computational compatibility)
        x_grid = np.stack([x_grid] * batch_size, axis=0)
        y_grid = np.stack([y_grid] * batch_size, axis=0)
    return x_grid, y_grid


def _dw_xcorr(phi_x, phi_z):
    """
    Implements the Depth-wise Cross Correlation (DW-XCorr) layer, as described in SiamRPN++ paper.
    For more info, refer to: <https://ieeexplore.ieee.org/document/8954116>
    :param phi_x: the embedding (feature map) of the srch_window region input
    :param phi_z: the embedding (feature map) of the temp_window region input
    :return: the depth-wise cross correlation of phi_x and phi_z.
    """
    mbatches_count = phi_x.shape[0]
    channels_count = phi_x.shape[1]

    corr_in_channels_count = mbatches_count * channels_count
    corr_out_channels_count = corr_in_channels_count

    phi_x = phi_x.view(1, corr_in_channels_count, phi_x.shape[2], phi_x.shape[3])
    phi_z = phi_z.view(corr_out_channels_count, 1, phi_z.shape[2], phi_z.shape[3])

    r = conv2d(input=phi_x, weight=phi_z, groups=corr_in_channels_count)

    r_h, r_w = r.shape[2], r.shape[3]

    # reshape back to dim=0 <=> batches
    r = r.view(mbatches_count, channels_count, r_h, r_w)
    return r


def create_video_from_frames(frames_path, output_video_path, fps=30):
    frame_files = sorted(os.listdir(frames_path))  # Assuming frames are named in sequential order
    print("equi_frame files: ", frame_files)
    if not frame_files:
        print("Error: No frames found in the specified directory.")
        return

    frame = cv2.imread(os.path.join(frames_path, frame_files[0]))
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use other codecs like 'MJPG' or 'XVID'

    output_video_path = output_video_path + '.mp4'
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame_file in frame_files:
        frame = cv2.imread(os.path.join(frames_path, frame_file))
        print(f"writing equi_frame file {frame_file}")
        video_writer.write(frame)

    video_writer.release()


def _callback_draw_bbox(event, x, y, flags, param):
    """
    Mouse callback function that handles mouse events for drawing bounding boxes.

    Callback Parameters (automatically filled):
        event: An integer holding the type of mouse event (e.g., mouse button click, mouse movement).
        x: The x-coordinate of the mouse cursor position when the event occurred.
        y: The y-coordinate of the mouse cursor position when the event occurred.
        flags: Additional flags passed by OpenCV (not used in this function).
        param: Additional parameters passed by OpenCV (not used in this function).

    Description of global variables:
        bbox_x: center of the fine_bbox (x-coordinate)
        bbox_y: center of the fine_bbox (y-coordinate)
        bbox_mn_w: width of the fine_bbox
        bbox_mn_h: height of the fine_bbox
        drawing_box: True if the fine_bbox is being drawn
        frame_bbox: equi_frame with the last fine_bbox drawn by the user
        equi_frame: original input equi_frame
    """
    # Description of global variables
    # bounding box center x, y, width, height, flag indicating the fine_bbox is being drawn, equi_frame containing the
    # drawn fine_bbox, original input equi_frame (without drawings), respectively
    global bbox_x, bbox_y, bbox_w, bbox_h, drawing_box, frame_bbox, frame

    if event == cv2.EVENT_LBUTTONDOWN:  # triggered while drawing (left button down)
        bbox_x, bbox_y = x, y
        drawing_box = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing_box:  # True if the 1) mouse is moving and 2) its left button is down
            frame_bbox = frame.copy()  # copying the original equi_frame has the effect of deleting the previously drawn
            # fine_bbox
            bbox_w, bbox_h = x - bbox_x, y - bbox_y
            cv2.rectangle(frame_bbox, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), (0, 255, 0),
                          2)  # draw fine_bbox

    elif event == cv2.EVENT_LBUTTONUP:  # Triggered after finishing fine_bbox drawing (releasing left mouse button)
        drawing_box = False
        bbox_w, bbox_h = x - bbox_x, y - bbox_y
        # Ensure width and height are positive
        if bbox_w < 0:
            bbox_x += bbox_w
            bbox_w = abs(bbox_w)
        if bbox_h < 0:
            bbox_y += bbox_h
            bbox_h = abs(bbox_h)
        # Store the center coordinates
        center_x = bbox_x + bbox_w // 2
        center_y = bbox_y + bbox_h // 2
        print("--")
        print("Bounding box center (x, y):", center_x, center_y)
        print("Bounding box dimensions (w, h):", bbox_w, bbox_h)


def draw_bbox(cur_frame):
    """
    :param cur_frame: current frame
    :param equi_frame: input equi_frame on which the user will draw the fine_bbox
    """
    # Create a named window and set the mouse callback
    cv2.setMouseCallback("Frame", _callback_draw_bbox)
    cv2.namedWindow("Frame")
    frame = cur_frame
    bbox_x, bbox_y, bbox_w, bbox_h, frame_bbox = 0, 0, 0, 0, None

    frame_copy = frame.copy()
    drawing_box = False
    while True:
        # Display the equi_frame
        if frame_bbox is None:
            cv2.imshow("Frame", frame_copy)
        else:
            cv2.imshow("Frame", frame_bbox)
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord("v"):
            break
    cv2.destroyAllWindows()
    print("----")
    print("Validated fine_bbox:")
    print(f"Center (x,y) = ({bbox_x},{bbox_y})\nDimensions (w,h) = ({bbox_w},{bbox_h})")
    print("----")


def get_bbox_corners(bbox):
    """
    Converts fine_bbox from [x_center, y_center, width, height] to corners [x_top_left, y_top_left, x_bottom_right,
     y_bottom_right]. Useful for image cropping
    :param bbox: bounding box, given as a list of [x_center, y_center, width, height]
    :return: bbox_corners: [x_top_left, y_top_left, x_bottom_right, y_bottom_right]
    """
    x_center, y_center, width, height = bbox
    x_tl = x_center - width * 0.5
    y_tl = y_center - height * 0.5
    x_rb = x_tl + width
    y_rb = y_tl + height
    bbox_corners = [x_tl, y_tl, x_rb, y_rb]
    return bbox_corners


class TrainTimer:
    def __init__(self, train_loader):
        self.time_batch_start = None
        self.time_batch_end = None
        self.time_per_step = None
        self.time_per_step_sum = 0
        self.time_per_batch_avg = 0
        self.processed_batches_num = None  #

        self.total_batches = len(train_loader)
        self.total_training_time = None  # holds the updated time required for training from scratch
        self.training_time_remaining = None  # the time remaining to finish the training

    def update_timing(self, batch_idx, time_batch_start):
        self.time_batch_start = time_batch_start
        self.processed_batches_num = batch_idx + 1  # +1 due to zero indexing
        self.time_batch_end = time.time()
        self.time_per_step = time.time() - self.time_batch_start
        print(f"current batch took = {round(self.time_per_step, 2)} seconds")
        self.time_per_step_sum += self.time_per_step
        self.time_per_batch_avg = self.time_per_step_sum / self.processed_batches_num

        self.total_training_time = self.total_batches * self.time_per_batch_avg
        self.training_time_remaining = self.total_training_time - (self.processed_batches_num * self.time_per_batch_avg)
        # number of batches
        training_time_remaining_formatted = self.convert_time(self.training_time_remaining)
        return training_time_remaining_formatted

    def convert_time(self, total_seconds, print_msg="Training Time Remaining:"):
        """Converts 'total_seconds' to a human-readable format of months, weeks, days, hours, minutes, and seconds."""
        months = int(total_seconds // (365 * 24 * 60 * 60))  # Months (assuming 30 days/month)
        total_seconds -= months * (365 * 24 * 60 * 60)
        weeks = int(total_seconds // (7 * 24 * 60 * 60))
        total_seconds -= weeks * (7 * 24 * 60 * 60)
        days = int(total_seconds // (24 * 60 * 60))
        total_seconds -= days * (24 * 60 * 60)
        hours = int(total_seconds // (60 * 60))
        total_seconds -= hours * (60 * 60)
        minutes = int(total_seconds // 60)
        remaining_seconds = int(total_seconds % 60)

        time_components = []
        if months > 0:
            time_components.append(f"{months} month{'s' if months > 1 else ''}")
        if weeks > 0:
            time_components.append(f"{weeks} week{'s' if weeks > 1 else ''}")
        if days > 0:
            time_components.append(f"{days} day{'s' if days > 1 else ''}")
        if hours > 0:
            time_components.append(f"{hours} hour{'s' if hours > 1 else ''}")
        if minutes > 0:
            time_components.append(f"{minutes} minute{'s' if minutes > 1 else ''}")
        time_components.append(f"{remaining_seconds} second{'s' if remaining_seconds > 1 else ''}")

        # Print the formatted time
        converted_time = print_msg, ", ".join(time_components)
        print(converted_time)
        return converted_time
