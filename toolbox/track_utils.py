import cv2
from config import cfg
import math
import numpy as np
import torch
from ViewPortRenderer import ViewPortRenderer
import warnings


class TrackHelper:
    def __init__(self):
        self.bbox_cen_m = 0
        self.bbox_cen_n = 0
        self.bbox_mn_w = 0
        self.bbox_mn_h = 0
        self.drawing_box = False
        self.frame_bbox = None
        self.frame = None

        self.context_amount = cfg.TRACK.CONTEXT_AMOUNT
        self.temp_size = cfg.TRACK.EXEMPLAR_SIZE
        self.srch_size = cfg.TRACK.INSTANCE_SIZE
        # size of the temp_window on the original equi_frame size before resizing to self.temp_size
        self.temp_org_size = None
        # size of the srch_window on the original equi_frame size before resizing to self.srch_size
        self.srch_org_size = None

        # Initializing cropping corners (for srch_window and temp_window): min -> top-left corner, max -> bottom-right
        # corner. We use these corners to compute the padding required, not to window
        self.crop_x_min = 0
        self.crop_y_min = 0
        self.crop_x_max = 0
        self.crop_y_max = 0

        # Initializing cropping corners (for srch_window and temp_window): min -> top-left corner, max -> bottom-right
        # corner These corners are shifted ('shf') to window the window from the padded equi_frame.
        self.crop_x_min_shf = 0
        self.crop_y_min_shf = 0
        self.crop_x_max_shf = 0
        self.crop_y_max_shf = 0

        # warning messages
        self.warning_1 = "Expected 'prediction' dictionary input to have at least the following keys: 'loc', " \
                         "'cls' and 'cen'.\nPlease check these keys in the'forward' method of SiamCAR class in " \
                         "models.py "
        self.warning_2 = "Invalid input type: 'prediction' must be a dictionary with at least three keys: 'loc', " \
                         "'cen', and 'cls' "

    def _callback_draw_bbox(self, event, x, y, flags, param):
        """
        Mouse callback function for (self.user_draw_bbox()) that handles mouse events for drawing bounding boxes.
        Callback Parameters (automatically filled):
            event: An integer holding the type of mouse event (e.g., mouse button click, mouse movement).
            x: The x-coordinate of the mouse cursor position when the event occurred.
            y: The y-coordinate of the mouse cursor position when the event occurred.
            flags: Additional flags passed by OpenCV (not used in this function).
            param: Additional parameters passed by OpenCV (not used in this function).

        Description of global variables:
            bbox_x: center of the fine_bbox (x-coordinate)
            bbox_y: center of the fine_bbox (y-coordinate)
            bbox_w: width of the fine_bbox
            bbox_h: height of the fine_bbox
            drawing_box: True if the fine_bbox is being drawn
            frame_bbox: frame with the last fine_bbox drawn by the user
            frame: original input frame
        """
        # Description of global variables
        # bounding box center x, y, width, height, flag indicating the fine_bbox is being drawn, frame containing the
        # drawn fine_bbox, original input frame (without drawings), respectively

        if event == cv2.EVENT_LBUTTONDOWN:  # triggered if left mouse-button is clicked (triggered once only)
            self.bbox_x, self.bbox_y = x, y
            self.drawing_box = True

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing_box:  # True if the 1) mouse is moving and 2) its left button is down
                self.frame_bbox = self.frame.copy()  # copying the original frame has the effect of deleting the
                # previously drawn fine_bbox
                self.bbox_w, self.bbox_h = x - self.bbox_x, y - self.bbox_y
                cv2.rectangle(self.frame_bbox, (self.bbox_x, self.bbox_y),
                              (self.bbox_x + self.bbox_w, self.bbox_y + self.bbox_h), (0, 255, 0),
                              2)  # draw fine_bbox

        elif event == cv2.EVENT_LBUTTONUP:  # Triggered after finishing fine_bbox drawing (releasing left mouse button)
            self.drawing_box = False
            self.bbox_w, self.bbox_h = x - self.bbox_x, y - self.bbox_y
            # Ensure width and height are positive
            if self.bbox_w < 0:
                self.bbox_x += self.bbox_w
                self.bbox_w = abs(self.bbox_w)
            if self.bbox_h < 0:
                self.bbox_y += self.bbox_h
                self.bbox_h = abs(self.bbox_h)
            # Store the center coordinates
            self.bbox_cen_x = self.bbox_x + self.bbox_w // 2
            self.bbox_cen_y = self.bbox_y + self.bbox_h // 2
            print("--")
            print("Bounding box center (x, y):", self.bbox_cen_x, self.bbox_cen_y)
            print("Bounding box dimensions (w, h):", self.bbox_w, self.bbox_h)

    def user_draw_bbox(self, frame, winname="Draw a bbox"):
        """
        :param frame: input frame on which the user will draw the fine_bbox
        :param winname: window name
        :return: fine_bbox as a list [center_x, center_y, width, height]
        """
        # Create a named window and set the mouse callback
        self.frame_bbox = None
        self.frame = frame
        self.window_name = winname
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._callback_draw_bbox)
        self.frame_copy = self.frame.copy()

        self.drawing_box = False
        while True:
            # Display the frame
            if self.frame_bbox is None:
                cv2.imshow(self.window_name, self.frame_copy)
            else:
                cv2.imshow(self.window_name, self.frame_bbox)
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                cv2.destroyWindow(self.window_name)
                break
        # cv2.destroyAllWindows()
        self.bbox = [self.bbox_cen_x, self.bbox_cen_y, self.bbox_w, self.bbox_h]
        print(f"Validated fine_bbox = [center_x, center_y, width, height] = {self.bbox}")

        self.get_org_crop_size()
        print(
            "The validated fine_bbox has set as a reference for temp_window and srch_window cropping. The temp_window "
            "will be cropped only once and used during the entire tracking phase")

        return self.bbox

    def get_org_sz_inputs(self, bbox, context):
        """
        Computes the original size of the SiamCAR inputs, i.e., srch_window and temp_window inputs
        :return:
        """
        bbox_cen_x, bbox_cen_y, bbox_w, bbox_h = bbox
        term0 = context * (bbox_w + bbox_h)
        term1 = (bbox_w + term0)
        term2 = (bbox_h + term0)
        temp_org_sz = math.sqrt(term1 * term2)
        srch_org_sz = temp_org_sz * 2
        return srch_org_sz, temp_org_sz

    def draw_bbox_mn(self, equi_frame, Wvp=500, Hvp=500):
        self.img_mn, self.obj_view = self.select_prj_view(equi_frame, Wvp=Wvp, Hvp=Hvp)
        # draw the bbox on user-selected projection
        self.bbox_mn = self.user_draw_bbox(self.img_mn, winname="draw bbox 1/2")
        return self.bbox_mn

    def get_dynamic_projection(self, bbox_mn, show=False, fov=None, fov_max_chg=3, smooth_fov=True):
        """
        Computes the dynamic projection of the bounding box onto the spherical viewport, adjusting the field of view
        (FOV) and orientation (phi, theta)

        This method performs the following steps:
        1. Updates the bounding box center and size parameters based on the input bounding box.
        2. Calculates the size of the search window in UV coordinates and derives the dynamic FOV angle using the arctan
           function.
        3. Converts the bounding box center coordinates from image space (M, N) to spherical coordinates (phi, theta).
        4. Optionally limits the change in FOV, phi, and theta values based on previous dynamic view parameters.
        5. Renders the viewport with the calculated dynamic FOV and orientation.
        6. Returns a dictionary containing the viewport image, phi, theta, and FOV.

        :param bbox_mn: Bounding box in the format [cen_x, cen_y, w, h] used to compute the dynamic view.
        :param show: Boolean flag to indicate whether to display the rendered viewport.
        :param fov: Field of view specified by the user; if provided, it overrides the computed FOV.
        :param prv_dynamic_view: Previous dynamic view parameters used to limit the change in FOV, phi, and theta.
        :param fov_max_chg: Threshold value for limiting the change in FOV, phi, and theta.
        :return: A dictionary containing the rendered viewport ('viewport'), dynamic phi ('phi'), dynamic theta
        ('theta'), and dynamic FOV ('fov').
        """
        self.bbox_mn = bbox_mn
        self.bbox_cen_m, self.bbox_cen_n, _, _ = self.bbox_mn
        self.srch_sz_mn, _ = self.get_org_sz_inputs(self.bbox_mn, self.context_amount)
        self.srch_sz_uv = self.srch_sz_mn * self.renderer.n_to_v
        # using srch_sz_uv, infer the dynamic field of view dynamic_fov angle using arctan2
        self.dynamic_fov = math.degrees(2 * math.atan(self.srch_sz_uv * 0.5))
        # convert bbox center to phi,theta
        self.dynamic_phi, self.dynamic_theta = self.mn_to_phit_heta(self.bbox_cen_m, self.bbox_cen_n)
        print(f"Current phi: {self.dynamic_phi}, Current theta {self.dynamic_theta}, Current fov {fov}")
        # if prv_dynamic_view is not None and smooth:
        #     self.dynamic_phi = self.change_limiter(cur_val=self.dynamic_phi, prv_val=prv_dynamic_view['phi'], thr=thr)
        #     self.dynamic_theta = self.change_limiter(cur_val=self.dynamic_theta, prv_val=prv_dynamic_view['theta'],
        #                                              thr=thr)
        #     print(f"Limited phi: {self.dynamic_phi}, Limited theta {self.dynamic_theta}")

        if fov is not None:
            fov = self.change_limiter(cur_val=self.dynamic_fov, prv_val=fov, thr=1)
            self.dynamic_viewport = self.renderer.render_viewport(fov=fov,
                                                                  theta_c=-self.dynamic_theta,
                                                                  phi_c=self.dynamic_phi,
                                                                  show=show)
        else:
            self.dynamic_viewport = self.renderer.render_viewport(fov=self.dynamic_fov,
                                                                  theta_c=-self.dynamic_theta,
                                                                  phi_c=self.dynamic_phi,
                                                                  show=show)

        self.dynamic_view = {'viewport': self.dynamic_viewport,
                             'phi': self.dynamic_phi,
                             'theta': self.dynamic_theta,
                             'fov': self.dynamic_fov}
        return self.dynamic_view

    def change_limiter(self, cur_val, prv_val, thr):
        """
        Limits the change in the current value based on a threshold relative to the previous value.

        This method ensures that the current value does not change by more than a specified threshold thr compared to
        the previous value. If the change exceeds the threshold, the current value is adjusted to stay within the
        threshold range.

        :param cur_val: The current value to be limited.
        :param prv_val: The previous value to compare against.
        :param thr: The threshold for limiting the change.
        :return: The adjusted current value, limited to within the threshold range relative to the previous value.
        """
        if abs(cur_val - prv_val) <= thr:
            return cur_val
        elif cur_val < prv_val:
            return prv_val - thr
        elif cur_val > prv_val:
            return prv_val + thr
        else:
            raise Exception("Unexpected behavior for change limiter function")

    def crop_window(self, frame, bbox, choice):
        """
        Crops and returns the search window or template window image based on the provided choice. This method crops a
        specified window from the input frame according to the given bounding box (`bbox`) and choice. It handles
        padding of the input frame to ensure that the cropping window fits within the image boundaries and performs
        resizing to match the required window size. The method also saves intermediate results for debugging purposes.

        :param frame: The input image frame from which to crop the window.
        :param bbox: A list or tuple specifying the bounding box as [center_x, center_y, width, height].
        :param choice: Specifies whether to crop the "srch_window" or "temp_window". It determines which size and parameters to use for cropping.
        :return: A tuple containing:
            - `window_as_tensor`: The cropped window as a PyTorch tensor, with shape (1, Channels, Height, Width).
            - `window`: The cropped window as a NumPy array, with shape (Height, Width, Channels).
        """
        self.choice = choice
        if self.choice == "srch_window":
            # Crop window for srch_window
            self.crop_org_size = self.srch_org_size
            self.crop_size = self.srch_size
        elif self.choice == "temp_window":
            # Crop window for temp_window
            self.crop_org_size = self.temp_org_size
            self.crop_size = self.temp_size
        else:
            raise ValueError("Choice must be either 'srch_window' or 'temp_window'")
        self.bbox_cen_m, self.bbox_cen_n = bbox[0], bbox[1]  # get fine_bbox center
        self.frm_height, self.frm_width, _ = frame.shape
        self.frame = frame

        # pad based on the fine_bbox center and window size
        self.half_org_size = 0.5 * (self.crop_org_size - 1)
        self.crop_x_min = math.floor(self.bbox_cen_m - self.half_org_size)
        self.crop_y_min = math.floor(self.bbox_cen_n - self.half_org_size)
        self.crop_x_max = self.crop_x_min + self.crop_org_size
        self.crop_y_max = self.crop_y_min + self.crop_org_size

        self.left_pad = int(max(0, -self.crop_x_min))
        self.top_pad = int(max(0, -self.crop_y_min))
        self.right_pad = int(max(0, self.crop_x_max - (self.frm_width - 1)))
        self.bottom_pad = int(max(0, self.crop_y_max - (self.frm_height - 1)))

        self.frm_padded_width = int(self.frm_width + self.right_pad + self.left_pad)
        self.frm_padded_hight = int(self.frm_height + self.top_pad + self.bottom_pad)
        self.frm_padded = np.zeros((self.frm_padded_hight, self.frm_padded_width, 3), dtype=np.uint8)
        self.pad_bgr = np.mean(frame, axis=(0, 1))
        self.frm_padded[:, :, :] = self.pad_bgr

        # inserting the original frame into the padded frame
        self.insert_frm_x_min = self.left_pad
        self.insert_frm_x_max = self.left_pad + self.frm_width
        self.insert_frm_y_min = self.top_pad
        self.insert_frm_y_max = self.top_pad + self.frm_height
        self.frm_padded[self.insert_frm_y_min:self.insert_frm_y_max,
        self.insert_frm_x_min:self.insert_frm_x_max,
        :] = self.frame

        # shifting cropping coordinates to window from the padded frame
        self.bbox_cen_x_shf = self.bbox_cen_m + self.left_pad
        self.bbox_cen_y_shf = self.bbox_cen_n + self.top_pad

        self.crop_x_min_shf = int(math.floor(self.bbox_cen_x_shf - self.half_org_size))
        self.crop_y_min_shf = int(math.floor(self.bbox_cen_y_shf - self.half_org_size))
        self.crop_y_max_shf = int(self.crop_y_min_shf + self.crop_org_size)
        self.crop_x_max_shf = int(self.crop_x_min_shf + self.crop_org_size)

        self.crop_org = self.frm_padded[self.crop_y_min_shf:self.crop_y_max_shf,
                        self.crop_x_min_shf:self.crop_x_max_shf,
                        :]
        self.window = cv2.resize(self.crop_org, (self.crop_size, self.crop_size))

        self.window_as_tensor = torch.from_numpy(self.window.astype(np.float32))
        # org is 0: H, 1: W, 2:Ch, we want Ch, H, W
        self.window_as_tensor = self.window_as_tensor.permute(2, 0, 1)  # Ch:2, H: 0, W: 1
        self.window_as_tensor = self.window_as_tensor.unsqueeze(0)  # add a batch dimension as the first dim
        return self.window_as_tensor, self.window

    def get_siamcar_inputs_360(self, viewport, choice=None):
        """
        Gets the projected input images (srch_window and/or temp_window) for SiamCAR.
        :param choice: possible options are "srch_window" or "temp_window". If None, both srch_window and temp_window
        images will be returned.
        :return: a dictionary containing the srch_window and/or temp_window images, resized for input compatibility.
        Possible keys are: srch_window, srch_tensor, temp_window, temp_tensor. keys with "window" are numpy images,
        whereas keys with "tensor" are the numpy images converted to torch tensors, then processed for compatibility
        with the model's input
        """
        if viewport.shape[0] != viewport.shape[1]:
            raise ValueError("viewport argument must be a square image, i.e., Width = Height")
        viewport_sz = viewport.shape[0]

        def compute_search(viewport):
            search = cv2.resize(viewport, (cfg.TRAIN.SEARCH_SIZE, cfg.TRAIN.SEARCH_SIZE))
            return search

        def compute_template(viewport, viewport_sz):
            temp_org_sz = int(viewport_sz / 2)  # also the center of the viewport image
            template = viewport[temp_org_sz - int(temp_org_sz / 2):temp_org_sz + int(temp_org_sz / 2),
                       temp_org_sz - int(temp_org_sz / 2):temp_org_sz + int(temp_org_sz / 2), :]
            template = cv2.resize(template, (cfg.TRAIN.EXEMPLAR_SIZE, cfg.TRAIN.EXEMPLAR_SIZE))
            return template

        def img_to_tensor(img):
            img_as_tensor = torch.from_numpy(img.astype(np.float32))
            # org is 0: H, 1: W, 2:Ch, we want Ch, H, W
            img_as_tensor = img_as_tensor.permute(2, 0, 1)  # Ch:2, H: 0, W: 1
            img_as_tensor = img_as_tensor.unsqueeze(0)  # add a batch dimension as the first dim
            return img_as_tensor

        viewports_dict = {
            'srch_window': None,
            'srch_tensor': None,
            'temp_window': None,
            'temp_tensor': None
        }

        if choice == "srch_window":
            self.srch_window = compute_search(viewport)
            self.srch_tnsr = img_to_tensor(self.srch_window)
            viewports_dict['srch_window'] = self.srch_window
            viewports_dict['srch_tensor'] = self.srch_tnsr

        elif choice == "temp_window":
            self.temp_window = compute_template(viewport, viewport_sz)
            self.temp_tnsr = img_to_tensor(self.temp_window)
            viewports_dict['temp_window'] = self.temp_window
            viewports_dict['temp_tensor'] = self.temp_tnsr

        elif choice is None:
            self.srch_window = compute_search(viewport)
            self.srch_tnsr = img_to_tensor(self.srch_window)
            self.temp_window = compute_template(viewport, viewport_sz)
            self.temp_tnsr = img_to_tensor(self.temp_window)
            viewports_dict['srch_window'] = self.srch_window
            viewports_dict['srch_tensor'] = self.srch_tnsr
            viewports_dict['temp_window'] = self.temp_window
            viewports_dict['temp_tensor'] = self.temp_tnsr
        else:
            raise ValueError("invalid choice argument. Consult the method's description")

        return viewports_dict

    def adapt_bbox(self, bbox, original_size, new_size, center=False):
        """
        Resizes a bounding box from the original image size to a new image size.

        :param bbox: A tuple or list containing the bounding box parameters (center_x, center_y, width, height).
        :param original_size: A tuple containing the original dimensions (height, width) of the image.
        :param new_size: A tuple containing the new dimensions (height, width) to resize the image to.
        :return: A tuple containing the resized bounding box (new_center_x, new_center_y, new_width, new_height).
        """
        bbox_cen_m, bbox_cen_n, bbox_mn_w, bbox_mn_h = bbox
        Hvp, Wvp = original_size
        new_width, new_height = new_size

        # Compute the scaling factors
        scale_x = new_width / Wvp
        scale_y = new_height / Hvp

        # Adapt the bounding box parameters
        if center:
            adapted_bbox_cen_m = new_width/2
            adapted_bbox_cen_n = new_height/2
        else:
            adapted_bbox_cen_m = bbox_cen_m * scale_x
            adapted_bbox_cen_n = bbox_cen_n * scale_y
        adapted_bbox_mn_w = bbox_mn_w * scale_x
        adapted_bbox_mn_h = bbox_mn_h * scale_y
        adapted_bbox = [int(adapted_bbox_cen_m), int(adapted_bbox_cen_n), int(adapted_bbox_mn_w), int(adapted_bbox_mn_h)]
        if center:
            print("Adapted bounding box (center_x, center_y, width, height):", adapted_bbox)
            print("Note: the bbox center has been set to the image center")
        else:
            print("Adapted bounding box (center_x, center_y, width, height):", adapted_bbox)

        return adapted_bbox

    def get_org_crop_size(self):
        """
        Use the initialized fine_bbox to compute the srch_window and temp_window window size on the original equi_frame size
        :return:
        """
        temp_org_w = self.bbox_w + self.context_amount * (self.bbox_w + self.bbox_h)
        temp_org_h = self.bbox_h + self.context_amount * (self.bbox_w + self.bbox_h)
        self.temp_org_size = round(math.sqrt(temp_org_w * temp_org_h))
        self.srch_org_size = self.temp_org_size * (self.srch_size/self.temp_size)

    def overlay_bbox(self, image, bbox, winname="FRAME", upscale_factor=None, show=True, thickness=4):
        # Extract center coordinates and dimensions
        cen_x, cen_y, width, height = bbox

        # Calculate the top-left and bottom-right coordinates of the bounding box
        top_left_x = int(cen_x - width / 2)
        top_left_y = int(cen_y - height / 2)
        bottom_right_x = int(cen_x + width / 2)
        bottom_right_y = int(cen_y + height / 2)

        # Draw the bounding box on the image
        cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), thickness)
        # Display the image with the bounding box
        h, w, _ = image.shape
        if upscale_factor is not None:
            image = cv2.resize(image, (int(w * upscale_factor), int(h * upscale_factor)))
        if show:
            cv2.imshow(winname, image)
            cv2.waitKey(0)
            cv2.destroyWindow(winname)
        return image

    def post_process_logits(self, prediction):
        """
        Postprocessing model's output logits using: 1) softmax for 'cls' tensor, and 2) sigmoid for 'cen' tensor.
        :param prediction: prediction dictionary with at least three keys 'loc', 'cen', and 'cls'
        :return: predictions with post-processed logits if input is valid, otherwise, return the same input without post
        -processing
        """
        if isinstance(prediction, dict):
            # Check if the required keys are present
            if 'loc' in prediction and 'cls' in prediction and 'cen' in prediction:
                prediction['cls'] = torch.softmax(prediction['cls'], dim=0)
                prediction['cen'] = torch.sigmoid(prediction['cen'])
            else:
                raise Warning(self.warning_1)
        else:
            raise Warning(self.warning_2)
        return prediction

    def compute_penalty_ij(self, cur_pred, prv_pred, k_pen):
        """
        Compute the penalty for each point ij, as described in SiamRPN paper, equation 13
        :param cur_pred: current prediction dictionary (squeezed for batch dim)
        :param prv_pred: previous prediction dictionary (squeezed for batch dim)
        :param k_pen: hyper-parameter 'k' for controlling the penalty (see equation 13 in SiamRPN paper)
        :return:
        """
        predictions = [cur_pred, prv_pred]
        for pred in predictions:
            if isinstance(pred, dict):
                # Check if the required keys are present
                if 'loc' in pred and 'cls' in pred and 'cen' in pred:
                    pass
                else:
                    raise Warning(self.warning_1)
            else:
                raise Warning(self.warning_2)

        self.cur_ratio_ij, self.cur_size_ij = self.compute_size_and_ratio_ij(cur_pred)
        self.prv_ratio_ij, self.prv_size_ij = self.compute_size_and_ratio_ij(prv_pred)

        self.ratio_change_ij = torch.max(self.cur_ratio_ij / self.prv_ratio_ij, self.prv_ratio_ij / self.cur_ratio_ij)
        self.size_change_ij = torch.max(self.cur_size_ij / self.prv_size_ij, self.prv_size_ij / self.cur_size_ij)

        self.penalty_ij = torch.exp(k_pen * self.ratio_change_ij * self.size_change_ij)
        return self.penalty_ij

    def compute_size_and_ratio_ij(self, prediction):
        """
        Computes and returns:
        1) the height to width ratio at each spatial point ij
        2) the padded size at each point (as described in SiamRPN paper, equation 13)
        :param prediction: prediction dictionary (squeezed for batch dimension)
        """
        # prediction['loc'][u], left, top, right, bottom <= > u = 0, 1, 2, 3, respectively
        width_ij, height_ij = self.compute_width_height_ij(prediction)
        ratio_ij = height_ij / width_ij

        size_pad = 0.5 * (width_ij + height_ij)  # see equation 14 in SiamRPN paper
        size_ij = torch.sqrt((width_ij + size_pad) * (width_ij + height_ij))
        return ratio_ij, size_ij

    def compute_width_height_ij(self, prediction):
        width_ij = prediction['loc'][0] + prediction['loc'][2]
        height_ij = prediction['loc'][1] + prediction['loc'][3]
        return width_ij, height_ij

    def select_prj_view(self, frame, Wvp, Hvp):
        def empty(x):
            pass
        Window_Title = "Image Viewer"
        cv2.namedWindow(Window_Title)

        cv2.createTrackbar("In/Out", Window_Title, 90, 175, empty)
        cv2.createTrackbar("Down/Up", Window_Title, 90, 180, empty)
        cv2.createTrackbar("Left/Right", Window_Title, 180, 360, empty)

        self.renderer = ViewPortRenderer(equi_img=frame)

        self.ptheta = -1
        self.pphi = -1
        self.pFOV = -1

        while True:
            FOV = cv2.getTrackbarPos("In/Out", Window_Title)
            theta = cv2.getTrackbarPos("Down/Up", Window_Title) - 90
            phi = cv2.getTrackbarPos("Left/Right", Window_Title) - 180
            if theta != self.ptheta or phi != self.pphi or FOV != self.pFOV:
                print("theta = ", theta)
                print("phi = ", phi)
                self.ptheta = theta
                self.pphi = phi
                self.pFOV = FOV
                viewport = self.renderer.render_viewport(fov=FOV, theta_c=theta, phi_c=phi)
                cv2.imshow(Window_Title, viewport)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                obj_view = {"phi": self.pphi,
                            "theta": self.ptheta,
                            "fov": self.pFOV}
                cv2.destroyWindow(Window_Title)
                return viewport, obj_view

    def mn_to_phit_heta(self, m, n):
        """
        Converts (m, n) coordinates of the projected image into spherical coordinates (phi, theta), i.e., (longitude,
        latitude). The Range of phi and theta is [-180°, 180°] and [-90°, 90°], respectively, which is the range
        compatible with the viewport renderer class
        :param m: horizontal coordinate of the projected image
        :param n: vertical coordinate of the projected image
        :return:
        """
        x_equi = self.renderer.x_equi_int[n, m]
        y_equi = self.renderer.y_equi_int[n, m]
        phi = (x_equi / (self.renderer.W_equi - 1) - 0.5) * 360
        theta = (y_equi / (self.renderer.H_equi - 1) - 0.5) * 180
        return phi, theta

    def interpolate_bbox_to_higher_resolution(self, bbox, original_dims, target_dims):
        """
        Interpolates a bounding box from original image dimensions to a higher resolution target image dimensions using bilinear interpolation.

        Parameters:
        - bbox: List of [bbox_cen_x, bbox_cen_y, bbox_w, bbox_h]
        - original_dims: Tuple of (original_width, original_height)
        - target_dims: Tuple of (target_width, target_height)

        Returns:
        - interpolated_bbox: List of [new_bbox_cen_x, new_bbox_cen_y, new_bbox_w, new_bbox_h]
        """
        bbox_cen_x, bbox_cen_y, bbox_w, bbox_h = bbox
        original_width, original_height = original_dims
        target_width, target_height = target_dims

        # Calculate the scaling factors
        scale_x = target_width / original_width
        scale_y = target_height / original_height


        # Interpolate the bounding box center coordinates and dimensions
        new_bbox_cen_x = self.interpolate(bbox_cen_x, scale_x, original_width, target_width)
        new_bbox_cen_y = self.interpolate(bbox_cen_y, scale_y, original_height, target_height)
        new_bbox_w = self.interpolate(bbox_w, scale_x, original_width, target_width)
        new_bbox_h = self.interpolate(bbox_h, scale_y, original_height, target_height)

        return [new_bbox_cen_x, new_bbox_cen_y, new_bbox_w, new_bbox_h]

    def interpolate(self, coord, scale, original_dim, target_dim):
        """
        Interpolate a coordinate using bilinear interpolation.

        Parameters:
        - coord: Coordinate in original image
        - scale: Scale factor
        - original_dim: Original dimension size
        - target_dim: Target dimension size

        Returns:
        - Interpolated coordinate in target image
        """
        new_coord = coord * scale
        lower_coord = np.floor(new_coord).astype(int)
        upper_coord = np.ceil(new_coord).astype(int)

        if lower_coord == upper_coord:
            return new_coord

        lower_weight = (upper_coord - new_coord) / (upper_coord - lower_coord)
        upper_weight = (new_coord - lower_coord) / (upper_coord - lower_coord)

        interpolated_coord = lower_coord * lower_weight + upper_coord * upper_weight
        return interpolated_coord
