import torch
from models.models import SiamCarV1
from config import cfg
from toolbox.track_utils import TrackHelper
from toolbox.misc import compute_xy_grid, get_bbox_corners
import os
import numpy as np
import cv2
from ViewPortRenderer import ViewPortRenderer
import subprocess
import imageio_ffmpeg as ffmpeg


class TrackerHead360:
    """
    Tracker head for object tracking uisng SiamCAR predictions
    """
    def __init__(self, model_path, lambda_d=0.5, k_pen=0.04):
        """
        Initialize the tracker360. Required initializations are:
        1) Get the trained model from 'model_path' and set it to evaluation mode
        2) Initialize hyper-paramters: lambda_d (see equation 9), k_pen (SiamRPN paper, k in equation 13)
        3) Get the first video equi_frame, and the first fine_bbox as (x_center, y_center, width, height)
        4) Extract the template image 'temp_img' (only once, for off-line tracking) using the fine bbox
        5) Extract the first srch_window image 'srch_img' (required to compute LTRB tensor, which will be used during
        the next prediction for penalty computation)
        6) Create a method in SiamCAR class called 'compute_template_branch', and compute it using 'temp_img' (its
        output will be fixed and used during the entire tracking)
        7) Get LTRB for reference of the next penalty.
        8) compute the hanning map (only once)
        9) update the ratio r_f0 and size s_f0. 0 means previous, 1 will mean current equi_frame: required for penalty value
        :param model_path: path to the SiamCAR model
        :param lambda_d: balance weight (see equation 9), set as in original paper
        :param k_pen: hyper-parameter controling the penalty p_ij (Denoted as k in SiamRPN paper, equation 13)
        """
        # A 2D numpy array containing all x (for x_grid) and y (for y_grid) coordinates of the srch_window region,
        # starting from 0 to search_width - 1 , that correspond to i coordinates of the output map.
        self.x_grid, self.y_grid = compute_xy_grid(cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.SEARCH_SIZE)
        self.srch_x_vals, self.srch_y_vals = self.x_grid[0, :], self.y_grid[:, 0]
        self.lambda_d = lambda_d  # balance weight (see equation 9), set as in original paper
        self.k_pen = k_pen  # tracking penalty
        self.model = SiamCarV1(mode='track_2d')  # Instantiate the tracking model and set it to tracking mode
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint['model'])  # Load the trained model
            print(f"Loaded model from: {model_path}")
        self.model.eval()  # set the model to test mode
        self.compute_cos_window_ij(cfg.TRACK.SCORE_SIZE)  # create the hanning function (once only)
        self.track_helper = TrackHelper()  # helper object to access useful tools
        self.bbox_ab = None  # coordinates of the bbox within the resized search image [x_cen, y_cen, width, height]
        self.end_idx = None
        print("TrackerHead360 instantiated")

    def compute_cos_window_ij(self, m):
        """
        Computes the hanning window or map. The hanning map returned assumes the model output maps have the same spatial
        size and that the width of each map is equal to the hight (e.g., 25x25)
        :param m: an integer scalar, specifying the width or height of the output map.
        :return: the hanning map, denoted as Hij in the original paper (see equation 9)
        """
        han_1d = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(m) / (m - 1))
        # Outer product to get 2D Hanning window
        self.cos_window_ij = np.outer(han_1d, han_1d)
        return self.cos_window_ij

    def squeeze_batch_dim(self, prediction):
        """
        Removes the batch dimension (e.g., prediction['loc'].shape = (1, 4, 26, 26) will become (4, 26, 26)
        :param prediction: the model's output prediction dictionary
        :return: prediction dictionary where all values have no batch dimension
        """
        self.prediction_keys = list(prediction.keys())
        for key in self.prediction_keys:
            prediction[key] = prediction[key].squeeze(0)
        return prediction

    def track_360(self, video_path, start_idx=0, save_video=True, ar_tag='', end_idx=None, viewport_size=1500,
                  smooth_bbox=True, bbox_smooth_thr=1, bbox_thr_factor=[7, 4], dst_dir=None,
                  adapt_fov=False, smooth_fov=True, fov_max_chg=3):
        """

        Threshold value to limit changes in bounding box coordinates.

        Implements the tracking loop in a 360-degree video using SiamCAR predictions.
        :param video_path: Path to the input 360-degree video file.
        :param start_idx: Index of the initial frame to start tracking from.
        :param save_video: Boolean flag indicating whether to save the output video.
        :param ar_tag: Augmented reality tag (text) to overlay on the viewport center. This text is backprojected on the
                       original 360° video. Ignored if left empty string ''.
        :param end_idx: Index of the frame to stop tracking.
        :param viewport_size: width and height of the projected viewport window centering the target
        :param smooth_bbox: enables smooth track if True (default). The smoothing
        :param bbox_smooth_thr:  smoothing threshold applied on bbox center and size (i.e. with and height) during the
                                 next projection, multiplied by thr_factor[0] and thr_factor[1] to get center threshold
                                 and width and height threshold, respectively.
        :param bbox_thr_factor: Optional list of factors to multiply the threshold for (cen_x, cen_y) and (w, h)
                                respectively. Default is [7, 4].
        :param dst_dir: video output directory. Defaults to 'results'
        :param adapt_fov: if True, the next viewport is projected using dynamic fov computed from the previous bbox
        :param smooth_fov: enables, if True, the fov smoothing by clipping values with changes higher than fov_max_chg.
        :param fov_max_chg: max change allowed for the new fov. Discarded if smooth_fov = False.
        :return: None
        """
        self.ar_tag = ar_tag
        self.viewport_and_bbox = None
        self.frame_idx = start_idx
        self.video_path = video_path

        self.equi_frame = self.read_equi_frame(video_path, self.frame_idx)
        self.equi_h, self.equi_w, _ = self.equi_frame.shape

        self.total_frames = int(self.equi_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.end_idx = self.total_frames if end_idx is None else end_idx

        self.ffmpeg_path = ffmpeg.get_ffmpeg_exe()

        self.dst_dir = dst_dir
        if self.dst_dir is None:
            self.dst_dir = os.path.join(os.getcwd(), "results")
        if not os.path.exists(self.dst_dir):
            os.mkdir(self.dst_dir)

        self.out_path_360 = os.path.join(self.dst_dir, '360_result_' + os.path.basename(self.video_path))
        self.ffmpeg_command = [
            self.ffmpeg_path,
            '-y',  # Overwrite output file if it exists
            '-f', 'rawvideo',  # Input format
            '-vcodec', 'rawvideo',  # Input codec
            '-s', f'{self.equi_w}x{self.equi_h}',  # Input resolution
            '-pix_fmt', 'bgr24',  # Input pixel format
            '-r', str(30),  # Input frame rate
            '-i', '-',  # Input from stdin
            '-c:v', 'libx264',  # Output codec
            '-preset', 'slow',  # Preset for better quality
            '-crf', '18',  # Constant Rate Factor for quality
            '-profile:v', 'high',  # High profile for better quality
            '-pix_fmt', 'yuv420p',  # Output pixel format
            self.out_path_360  # Output file
        ]
        self.process = subprocess.Popen(self.ffmpeg_command, stdin=subprocess.PIPE)

        self.init_tracker_360(self.video_path, self.frame_idx)
        self.high_res_renderer = ViewPortRenderer(self.equi_frame, viewport_size, viewport_size)
        self.srch_org_size = self.track_helper.renderer.Wvp
        self.srch_scale_factor = self.srch_org_size / self.track_helper.srch_size
        self.save_video = save_video
        if self.save_video:
            self.create_video_writer(file_prefix="viewport_", viewport_mode=True)
        self.track_result_winname = "Tracking Viewport"
        cv2.namedWindow(self.track_result_winname)

        while True:  # tracking loop
            self.ret, self.equi_frame = self.equi_cap.read()  # get a frame
            current_frame_index = int(self.equi_cap.get(cv2.CAP_PROP_POS_FRAMES))
            if not self.ret:
                print("Breaking the tracking loop: could not read the equi frame")
                break  # Break if no more frames
            if current_frame_index > self.end_idx:
                break  # Break if ending index is reached
            self.track_helper.renderer.set_equi_img(self.equi_frame)
            if adapt_fov:
                self.dynamic_fov = self.dynamic_view['fov']

                self.dynamic_view = self.track_helper.get_dynamic_projection(self.bbox_mn,
                                                                             show=False,
                                                                             fov=self.dynamic_fov,
                                                                             fov_max_chg=fov_max_chg,
                                                                             smooth_fov=smooth_fov)  # self.dynamic_view,
                # none will not apply a change limiter
            else:
                self.dynamic_view = self.track_helper.get_dynamic_projection(self.bbox_mn,
                                                                             show=False,
                                                                             fov=self.initial_fov,
                                                                             fov_max_chg=fov_max_chg,
                                                                             smooth_fov=smooth_fov)  # self.dynamic_view,
                # none will not apply a change limiter

            self.siamcar_inputs = self.track_helper.get_siamcar_inputs_360(self.dynamic_view['viewport'],
                                                                           choice="srch_window")

            self.cur_pred = self.compute_siamcar_prediction(x=self.siamcar_inputs['srch_tensor'],
                                                            z=None)  # z=None: temp branch is pre-computed

            # Compute equation 9 to get coarse location (i,j), then get corresponding (x, y, w, h)
            self.penalty_ij = self.track_helper.compute_penalty_ij(cur_pred=self.cur_pred,
                                                                   prv_pred=self.prv_pred,
                                                                   k_pen=self.k_pen)

            self.target_xy_cen_srch = self.coarse_localizer(lambda_d=self.lambda_d, cls_ij=self.cur_pred['cls'],
                                                            penalty_ij=self.penalty_ij, cen_ij=self.cur_pred['cen'],
                                                            cos_window_ij=self.cos_window_ij)
            # get target_xy_loc_frame : location on the frame
            self.coarse_bbox = self.get_coarse_bbox(self.target_xy_cen_srch, self.cur_pred)
            # modify the fine bbox such that it is represented within projected image (m,n) coordinate
            self.bbox_mn = self.get_bbox_mn(self.coarse_bbox, self.cur_pred, self.bbox_mn, smooth=smooth_bbox,
                                            thr=bbox_smooth_thr, thr_factor=bbox_thr_factor)
            # Draw the fine_bbox on the current search image
            self.viewport_and_bbox = self.track_helper.overlay_bbox(image=self.dynamic_view['viewport'],
                                                                    bbox=self.bbox_mn,
                                                                    show=False)
            # Update previous prediction with the current one
            self.prv_pred = self.cur_pred

            cv2.imshow(self.track_result_winname, self.viewport_and_bbox)
            self.video_writer.write(self.viewport_and_bbox)

            self.high_res_renderer.set_equi_img(new_equi_img=self.equi_frame)
            self.high_res_viewport = self.high_res_renderer.render_viewport(fov=self.initial_fov,
                                                                            theta_c=-self.dynamic_view['theta'],
                                                                            phi_c=self.dynamic_view['phi'])
            if self.ar_tag:
                self.high_res_viewport = self.high_res_renderer.add_centered_text(self.ar_tag, self.high_res_viewport)
            self.high_res_bbox = self.track_helper.interpolate_bbox_to_higher_resolution(bbox=self.bbox_mn,
                                                                                         original_dims=(
                                                                                         self.track_helper.renderer.Wvp,
                                                                                         self.track_helper.renderer.Wvp
                                                                                         ),
                                                                                         target_dims=(
                                                                                         self.high_res_renderer.Wvp,
                                                                                         self.high_res_renderer.Wvp))
            self.high_res_viewport = self.track_helper.overlay_bbox(image=self.high_res_viewport,
                                                                    bbox=self.high_res_bbox,
                                                                    show=False,
                                                                    thickness=12)
            self.equi_img_tag = self.high_res_renderer.remap_viewport_to_equirectangular(self.high_res_viewport)

            self.process.stdin.write(self.equi_img_tag.tobytes())
            # Wait for the specified amount of time to achieve the desired frame rate
            if cv2.waitKey(int(1000 / 30)) & 0xFF == ord('q'):
                cv2.destroyWindow(self.track_result_winname)
                break
        cv2.destroyAllWindows()
        self.equi_cap.release()
        self.video_writer.release()
        self.process.stdin.close()
        self.process.wait()


    def track_2d(self, video_path, frame_idx=0, save_video=True, dst_dir=None):
        """
        Coarse to fine tracking
        :return:
        """
        self.dst_dir = dst_dir
        if self.dst_dir is None:
            self.dst_dir = os.path.join(os.getcwd(), "results")
        self.frame_idx = frame_idx
        self.video_path = video_path
        self.init_tracker_2d(self.video_path)
        self.srch_scale_factor = self.track_helper.srch_org_size / self.track_helper.srch_size

        self.save_video = save_video
        if self.save_video:
            self.create_video_writer(file_prefix="2dTrk_", viewport_mode=False)
        self.prv_frame = self.frame  # update previous frame

        while True:  # tracking loop
            self.ret, self.frame = self.cap.read()  # get a frame
            if not self.ret:
                break  # Break if no more frames
            # Use the fine_bbox position from the previous frame to extract the search image from the current frame
            self.srch_tnsr, _ = self.track_helper.crop_window(self.frame, self.fine_bbox, "srch_window")
            # Use the search frame to get predictions. Use the current and previous ltrb to predict p_ij penalty map
            self.cur_pred = self.compute_siamcar_prediction(x=self.srch_tnsr,
                                                            z=None)  # z=None: temp branch is pre-computed
            # Compute equation 9 to get coarse location (i,j), then get corresponding (x, y, w, h)
            self.penalty_ij = self.track_helper.compute_penalty_ij(cur_pred=self.cur_pred, prv_pred=self.prv_pred,
                                                                   k_pen=self.k_pen)

            self.target_xy_cen_srch = self.coarse_localizer(lambda_d=self.lambda_d, cls_ij=self.cur_pred['cls'],
                                                            penalty_ij=self.penalty_ij, cen_ij=self.cur_pred['cen'],
                                                            cos_window_ij=self.cos_window_ij)
            # get target_xy_loc_frame : location on the frame
            self.coarse_bbox = self.get_coarse_bbox(self.target_xy_cen_srch, self.cur_pred)
            self.fine_bbox = self.get_fine_bbox(self.coarse_bbox, self.cur_pred)

            # Draw the fine_bbox on the current search image

            # Update previous prediction with the current one
            self.prv_pred = self.cur_pred
            # Use
            # To get the next search image (from the next frame), you must figure out the fine_bbox center on the original frame,
            # not the search frame.
            cv2.rectangle(self.frame, (self.fine_bbox[0] - self.fine_bbox[2] // 2, self.fine_bbox[1] - self.fine_bbox[3] // 2),
                          (self.fine_bbox[0] + self.fine_bbox[2] // 2, self.fine_bbox[1] + self.fine_bbox[3] // 2), (0, 255, 0), 2)

            cv2.imshow('Frame', self.frame)
            self.video_writer.write(self.frame)
            # Wait for the specified amount of time to achieve the desired frame rate
            if cv2.waitKey(int(1000 / 30)) & 0xFF == ord('q'):
                break
        self.cap.release()
        self.video_writer.release()

        cv2.destroyAllWindows()


    def create_video_writer(self, file_prefix='viewport', viewport_mode=True, fps=25):
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec (H264)
        self.video_out_file = file_prefix + os.path.basename(self.video_path)
        self.video_out_file = os.path.join(self.dst_dir, self.video_out_file)

        if viewport_mode:
            frm_sz = (int(self.srch_org_size), int(self.srch_org_size))
        else:
            frm_sz = (int(self.frm_width), int(self.frm_height))

        self.video_writer = cv2.VideoWriter(self.video_out_file, self.fourcc, fps, frm_sz)

    def get_fine_bbox(self, coarse_bbox, prediction):
        # .unsqueeze(0) adds a batch dimension for the interpolation API to work (expected input shape: batch_size,
        # channels, height, width)
        cen_ab = torch.nn.functional.interpolate(prediction['cen'].unsqueeze(0), size=(255, 255), mode='bicubic')
        loc_ab = torch.nn.functional.interpolate(prediction['loc'].unsqueeze(0), size=(255, 255), mode='bicubic')
        self.cen_ab = cen_ab.squeeze(0, 1)  # remove the batch and channel dimension (because we have one channel)
        self.loc_ab = loc_ab.squeeze(0)  # remove the batch dimension (but not the channels because we have 4)
        self.coarse_bbox_corners = get_bbox_corners(coarse_bbox)
        self.x_tl, self.y_tl, self.x_br, self.y_br = np.array(self.coarse_bbox_corners, dtype=np.int32)
        self.cen_gh = self.cen_ab[self.y_tl:self.y_br, self.x_tl:self.x_br]
        self.loc_gh = self.loc_ab[:, self.y_tl:self.y_br, self.x_tl:self.x_br]

        max_idx = torch.argmax(self.cen_gh)
        self.g_max = (max_idx % self.cen_gh.size(1)).item()  # col of the max value
        self.h_max = (max_idx // self.cen_gh.size(1)).item()  # row of the max value

        if self.cen_gh[self.h_max, self.g_max] != self.cen_gh.max():  # a little lunatic check up
            raise ValueError("Unexpected error: 'g_max' and 'h_max' do not correspond to the maximum value of cen_gh")

        self.a0 = self.g_max + self.x_tl
        self.b0 = self.h_max + self.y_tl

        self.ltrb_ab = self.loc_gh[:, self.h_max, self.g_max]

        self.width = (self.ltrb_ab[0] + self.ltrb_ab[2]) * self.srch_scale_factor
        self.height = (self.ltrb_ab[1] + self.ltrb_ab[3]) * self.srch_scale_factor

        self.u0, self.v0 = self.a0 * self.srch_scale_factor, self.b0 * self.srch_scale_factor
        self.m0, self.n0 = self.u0 + self.track_helper.crop_x_min_shf, self.v0 + self.track_helper.crop_y_min_shf
        self.x0, self.y0 = self.m0 - self.track_helper.left_pad, self.n0 - self.track_helper.top_pad

        self.fine_bbox = [int(self.x0), int(self.y0), int(self.width), int(self.height)]
        return self.fine_bbox

    def get_bbox_mn(self, coarse_bbox, prediction, prv_bbox_mn, smooth=True, thr=1, thr_factor=None):
        # .unsqueeze(0) adds a batch dimension for the interpolation API to work (expected input shape: batch_size,
        # channels, height, width)
        cen_ab = torch.nn.functional.interpolate(prediction['cen'].unsqueeze(0), size=(255, 255), mode='bicubic')
        loc_ab = torch.nn.functional.interpolate(prediction['loc'].unsqueeze(0), size=(255, 255), mode='bicubic')
        self.cen_ab = cen_ab.squeeze(0, 1)  # remove the batch and channel dimension (because we have one channel)
        self.loc_ab = loc_ab.squeeze(0)  # remove the batch dimension (but not the channels because we have 4)
        self.coarse_bbox_corners = get_bbox_corners(coarse_bbox)

        self.x_tl, self.y_tl, self.x_br, self.y_br = np.array(self.coarse_bbox_corners, dtype=np.int32)
        self.cen_gh = self.cen_ab[self.y_tl:self.y_br, self.x_tl:self.x_br]
        self.loc_gh = self.loc_ab[:, self.y_tl:self.y_br, self.x_tl:self.x_br]

        max_idx = torch.argmax(self.cen_gh)
        self.g_max = (max_idx % self.cen_gh.size(1)).item()  # col of the max value
        self.h_max = (max_idx // self.cen_gh.size(1)).item()  # row of the max value

        if self.cen_gh[self.h_max, self.g_max] != self.cen_gh.max():  # a little lunatic check up
            raise ValueError("Unexpected error: 'g_max' and 'h_max' do not correspond to the maximum value of cen_gh")

        self.a0 = self.g_max + self.x_tl
        self.b0 = self.h_max + self.y_tl

        self.ltrb_ab = self.loc_gh[:, self.h_max, self.g_max]

        self.width = ((self.ltrb_ab[0] + self.ltrb_ab[2]) * self.srch_scale_factor).detach()
        self.height = ((self.ltrb_ab[1] + self.ltrb_ab[3]) * self.srch_scale_factor).detach()

        self.u0, self.v0 = self.a0 * self.srch_scale_factor, self.b0 * self.srch_scale_factor

        if smooth:
            self.bbox_mn = self.limit_bbox_change(prv_bbox_mn, [self.u0, self.v0, self.width, self.height],
                                                  thr=thr, thr_factor=thr_factor)
        else:
            self.bbox_mn = [int(self.u0), int(self.v0), int(self.width),
                            int(self.height)]  # x_cen, y_cen, width, height
        return self.bbox_mn

    def limit_bbox_change(self, prv_bbox, cur_bbox, thr, thr_factor=[7, 4]):
        """
        Limit the change in bounding box coordinates between frames. This method constrains the amount of change allowed
        in the bounding box coordinates from the previous frame to the current frame to prevent large, abrupt changes
        that might result from inaccuracies or noise in the tracking process.

        :param prv_bbox: List of bounding box coordinates from the previous frame [cen_x, cen_y, w, h].
        :param cur_bbox: List of bounding box coordinates from the current frame [cen_x, cen_y, w, h].
        :param thr: Threshold value to limit changes in bounding box coordinates.
        :param thr_factor: Optional list of factors to multiply the threshold for (cen_x, cen_y) and (w, h) respectively.
                           Default is [7, 4].

        :return: List of bounding box coordinates after applying the change limit [cen_x, cen_y, w, h].
        """
        cen_fac, size_fac = thr_factor
        thr = np.array([thr * cen_fac, thr * cen_fac, thr * size_fac, thr * size_fac])
        prv_bbox = np.array(prv_bbox)
        cur_bbox = np.array(cur_bbox)
        abs_diff = np.abs(cur_bbox - prv_bbox)
        limited_bbox = prv_bbox + np.minimum(abs_diff, thr * (cur_bbox - prv_bbox) / abs_diff)
        return limited_bbox.astype(int).tolist()

    def compute_siamcar_prediction(self, x, z=None):
        """
        Compute the SiamCAR prediction without batch dimension and with logits post-processing.

        :param x: Input tensor for the search image.
        :param z: Input tensor for the template image (optional).
        :return: Processed prediction results from the SiamCAR model.
        """
        prediction = self.model(x, z)
        prediction = self.squeeze_batch_dim(prediction)
        prediction = self.track_helper.post_process_logits(prediction)
        return prediction

    def coarse_localizer(self, lambda_d, cls_ij, penalty_ij, cen_ij, cos_window_ij):
        cls_fg_ij = cls_ij[0].detach()
        cen_ij = cen_ij.detach().squeeze(0)
        penalty_ij = penalty_ij.detach()
        self.q = ((1 - lambda_d) * cls_fg_ij * penalty_ij * cen_ij) + (lambda_d * cos_window_ij)
        max_idx = torch.argmax(self.q)
        self.i_max = (max_idx % self.q.size(1)).item()  # col of the max value
        self.j_max = (max_idx // self.q.size(1)).item()  # row of the max value
        if self.q[self.j_max, self.i_max] != self.q.max():  # a little lunatic check up
            raise ValueError("Unexpected error: 'i_max' and 'j_max' do not correspond to the maximum value of q "
                             "(given in equation 9")
        target_xy_loc_srch = self.srch_x_vals[self.i_max], self.srch_y_vals[self.j_max]
        return target_xy_loc_srch

    def get_coarse_bbox(self, target_xy_loc_srch, prediction):
        """
        Get the fine_bbox [x, y, w, h] on the downscaled srch_window image coordinates
        :param target_xy_loc_srch:
        :param prediction:
        :return:
        """
        self.a0, self.b0 = target_xy_loc_srch
        # self.u0, self.v0 = self.a0 * self.srch_scale_factor, self.b0 * self.srch_scale_factor
        # self.m0, self.n0 = self.u0 + self.track_helper.crop_x_min_shf, self.v0 + self.track_helper.crop_y_min_shf
        # self.x0, self.y0 = self.m0 - self.track_helper.left_pad, self.n0 - self.track_helper.top_pad
        self.width_ij, self.height_ij = self.track_helper.compute_width_height_ij(prediction)
        # self.width_ij, self.height_ij = self.srch_scale_factor * self.width_ij, self.srch_scale_factor * self.height_ij
        self.width, self.height = self.width_ij[self.j_max, self.i_max], self.height_ij[self.j_max, self.i_max]
        # self.x0, self.y0, self.width, self.height = int(self.x0), int(self.y0), int(self.width), int(self.height)

        bbox = [self.a0, self.b0, self.width.item(), self.height.item()]
        # print(f"Target center xy ({self.x0}, {self.y0})\nTarget dims wh ({self.width}, {self.height})")
        return bbox

    def read_equi_frame(self, video_path, idx, release_cap=False):
        # Open the video file
        self.equi_cap = cv2.VideoCapture(video_path)
        # Check if video opened successfully
        if not self.equi_cap.isOpened():
            print("Error: Could not open video.")
            return None

        # Set the equi_frame position to the desired equi_frame index
        self.equi_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

        # Read the equi_frame at the current position
        ret, frame = self.equi_cap.read()

        # Release the video capture object
        if release_cap:
            self.equi_cap.release()

        # Check if the equi_frame was read successfully
        if not ret:
            print("Error: Could not read equi_frame.")
            return None

        return frame

    def init_tracker_360(self, video_path, idx):
        """
        Initialize the tracker with the specified video frame and setup necessary components.

        This method performs the following steps:
        1. Reads the specified frame from the video.
        2. Draws the initial bounding box on the frame.
        3. Recenters the object drawn and adapts the field of view for search region compatibility.
        4. Gets the dynamic projection for the bounding box.
        5. Extracts SiamCAR inputs from the dynamic view's viewport.
        6. Initializes the template branch of the SiamCAR model with the template tensor.
        7. Computes the initial predictions for the search tensor.
        8. Adapts the bounding box for search region compatibility
        9. Sets up the first prediction for tracking.

        :param video_path: Path to the video file.
        :param idx: Index of the frame to initialize the tracker with.
        """
        # reading the idx-th video frame
        self.equi_frame = self.read_equi_frame(video_path, idx)
        # Drawing the first bbox
        self.bbox_mn = self.track_helper.draw_bbox_mn(self.equi_frame)
        # recenter the first bbox and adapt the fov for search region compatiblity
        self.dynamic_view = self.track_helper.get_dynamic_projection(self.bbox_mn)
        # use the adapted view to get the inputs: downscaled search and template images
        self.siamcar_inputs = self.track_helper.get_siamcar_inputs_360(self.dynamic_view['viewport'], choice=None)
        # initialize the fov
        self.initial_fov = self.dynamic_view['fov']
        # pre-compute the template branch (offline tracking)
        self.model.initialize_temp_branch(z=self.siamcar_inputs['temp_tensor'])
        # update the previosu prediction
        self.prv_pred = self.compute_siamcar_prediction(x=self.siamcar_inputs['srch_tensor'],
                                                        z=self.siamcar_inputs['temp_tensor'])

        try:
            self.Hvp, self.Wvp, _ = self.dynamic_view['viewport'].shape
        except:
            raise ValueError("self.dynamic_viewport must be an numpy image with three channels")
        # redrawing the second bbox (on the adapted view: center and fov)
        self.bbox_mn = self.track_helper.user_draw_bbox(self.dynamic_view['viewport'], winname='draw bbox 2/2')
        # adapt the drawn bbox to suit the search image coordinates (a, b)
        self.bbox_ab = self.track_helper.adapt_bbox(self.bbox_mn, original_size=(self.Hvp, self.Wvp),
                                                    new_size=(cfg.TRACK.INSTANCE_SIZE, cfg.TRACK.INSTANCE_SIZE))
        print("The tracker has been initialized")


    def init_tracker_2d(self, video_path):
        """
        Makes the following initializations:
        1) draw and save the initial manual fine_bbox
        2) extract the search and template image
        2) initialize the template branch
        """
        self.fine_bbox = self.initialize_bbox(video_path=video_path)
        self.track_helper.get_org_crop_size()  # required before
        self.srch_tnsr, _ = self.track_helper.crop_window(self.frame, self.fine_bbox, 'srch_window')
        self.temp_tnsr, _ = self.track_helper.crop_window(self.frame, self.fine_bbox, 'temp_window')
        self.model.initialize_temp_branch(z=self.temp_tnsr)
        self.prv_pred = self.compute_siamcar_prediction(x=self.srch_tnsr, z=self.temp_tnsr)
        print("Tracker has been initialized: fine_bbox, template, search, first predictions\n--")


    def initialize_bbox(self, video_path):
        """
        Initializes the fine_bbox self.fine_bbox = [center_x, center_y, width, height]
        :param video_path:
        :return:
        """
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print("Error: Unable to open video file.")
            return
        # Get size of the video frames
        self.frm_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frm_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create a black image with the same shape
        self.black_image = np.zeros((self.frm_height, self.frm_width, 3), dtype=np.uint8)
        self.frame = self.black_image  # initialize the current frame to a black image
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
        self.ret, self.frame = self.cap.read()  # get the first frame
        bbox = self.track_helper.user_draw_bbox(self.frame)
        return bbox
