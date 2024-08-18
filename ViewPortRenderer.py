import numpy as np  # 1.23.4
import cv2


class ViewPortRenderer:
    def __init__(self, equi_img, Wvp=800, Hvp=800):
        """
        :param equi_img: input equirectangular image
        :param Wvp: Width of the viewport image
        :param Hvp: Height of the viewport image
        """
        self.equi_img = equi_img
        self.equi_img_meta = equi_img
        self.W_equi = self.equi_img.shape[1]  # width and height of the equirectangular image
        self.H_equi = self.equi_img.shape[0]
        self.equi_wcenter = self.W_equi / 2.0
        self.equi_hcenter = self.H_equi / 2.0

        self.Wvp = Wvp  # width and height of the viewport imageq
        self.Hvp = Hvp

        # establishing (m,n)
        self.m = np.linspace(0, self.Wvp - 1, self.Wvp)
        self.n = np.linspace(0, self.Hvp - 1, self.Hvp)

        self.z_axis = np.array([0.0, 0.0, 1.0], np.float32)  # rotation axis to change viewing longitude
        self.y_axis = np.array([0.0, 1.0, 0.0], np.float32)  # rotation axis to change viewing latitude
        self.Pt = np.ones((self.Hvp * self.Wvp, 3), np.float32)  # Matrix to hold triplets of spherical points to remap

        self.I = np.identity(3)

    def set_equi_img(self, new_equi_img):
        """
        Sets a new equirectangular image and updates related attributes.

        :param new_equi_img: new input equirectangular image
        """
        self.equi_img = new_equi_img
        self.equi_img_meta = new_equi_img
        self.W_equi = self.equi_img.shape[1]
        self.H_equi = self.equi_img.shape[0]
        self.equi_wcenter = self.W_equi / 2.0
        self.equi_hcenter = self.H_equi / 2.0

    def set_viewport_size(self, Wvp, Hvp):
        self.Wvp = Wvp  # width and height of the viewport imageq
        self.Hvp = Hvp

        # establishing (m,n)
        self.m = np.linspace(0, self.Wvp - 1, self.Wvp)
        self.n = np.linspace(0, self.Hvp - 1, self.Hvp)

        self.z_axis = np.array([0.0, 0.0, 1.0], np.float32)  # rotation axis to change viewing longitude
        self.y_axis = np.array([0.0, 1.0, 0.0], np.float32)  # rotation axis to change viewing latitude
        self.Pt = np.ones((self.Hvp * self.Wvp, 3), np.float32)  # Matrix to hold triplets of spherical points to remap

        self.I = np.identity(3)

    def render_viewport(self,
                        fov=120,
                        theta_c=0,
                        phi_c=0,
                        show=False):
        """
        Renders a View Port (VP) image from 360° images stored in equirectangular format, given a specific fov and
        viewing angle phi_c and theta_c
        :param fov: field of view in degrees
        :param theta_c: viewing latitude in degrees
        :param phi_c: viewing longitude in degrees
        :return:
        """
        self.phi_c = phi_c
        self.theta_c = -theta_c
        self.compute_uv(fov)
        self.compute_xyz()
        self.rotate_xyz()
        self.xyz_to_phi_theta()
        self.phi_theta_to_equi_xy()
        self.project_to_viewport()
        # self.remap_equi_xy_to_mn()  # just to get a better quality view port
        self.viewport = self.cubic_interpolation_vectorized(self.equi_img, self.x_equi, self.y_equi)
        if show:
            cv2.imshow("Rendered Viewport", self.viewport)
            cv2.waitKey()
            cv2.destroyAllWindows()
        return self.viewport

    def compute_uv(self, fov):
        self.fov_h = fov
        # self.fov_v = self.fov_h * (self.Wvp / self.Hvp)
        self.fov_v = self.fov_h * (float(self.Hvp) / float(self.Wvp))
        self.fov = (self.fov_h, self.fov_v)
        # compute all pairs  (m,n)
        self.Wuv = 2 * np.tan(np.radians(self.fov_h / 2))  # width of the UV plane
        self.Huv = 2 * np.tan(np.radians(self.fov_v / 2))  # height of the UV plane

        self.m_to_u = self.Wuv / self.Wvp
        self.n_to_v = self.Huv / self.Hvp
        self.u = self.m * self.m_to_u
        self.v = self.n * self.n_to_v
        return self.u, self.v

    def compute_xyz(self):
        """
        compute rotated Pt from uv
        :return:
        """
        self.x = np.ones(shape=(self.Hvp, self.Wvp))  # x = R = 1
        self.y = self.u - self.Wuv * 0.5
        self.z = -self.v + self.Huv * 0.5
        self.y, self.z = np.meshgrid(self.y, self.z)
        self.rd = np.sqrt((self.x ** 2) + (self.y ** 2) + (self.z ** 2))
        self.x = self.x / self.rd
        self.y = self.y / self.rd
        self.z = self.z / self.rd

    def rotate_xyz(self):
        # rotate around z-axis (longitude rotation)
        self.R_phi_c = self.rotation_mat(k=self.z_axis, alpha=self.phi_c)
        # rotate around y-axis (latitude rotation) by R_phi_c
        self.R_theta_c = self.rotation_mat(k=self.y_axis, alpha=self.theta_c)
        self.R = np.matmul(self.R_phi_c, self.R_theta_c)

        self.Pt[:, 0] = self.x.flatten()  # flattening by stacking rows
        self.Pt[:, 1] = self.y.flatten()
        self.Pt[:, 2] = self.z.flatten()
        self.Pt = self.Pt.T
        self.Pt = np.matmul(self.R, self.Pt)
        self.Pt = self.Pt.T

    def rotation_mat(self, k, alpha):
        """
        :param k: axis of rotation [kx, ky, kz], as a list, tuple, or 1d numpy array
        :param alpha: angle of rotation (in degrees)
        :return: Rodriguez 3x3 matrix
        """
        kx, ky, kz = k
        K = np.array([[0, -kz, ky],
                      [kz, 0, -kx],
                      [-ky, -kx, 0]])
        R = self.I + np.sin(np.radians(alpha)) * K + (1 - np.cos(np.radians(alpha))) * np.matmul(K, K)
        return R

    def xyz_to_phi_theta(self):
        """
        compute spherical coordinates (phi, theta) from Pt
        """
        # of shape (self.Wvp*self.Hvp,) ; order is important: must correspond to u and v values
        self.phi = np.arctan2(self.Pt[:, 1], self.Pt[:, 0])  # longitude
        self.theta = -np.arcsin(self.Pt[:, 2])  # latitude
        self.phi = np.degrees(self.phi.reshape([self.Hvp, self.Wvp]))
        self.theta = np.degrees(self.theta.reshape([self.Hvp, self.Wvp]))
        return self.phi, self.theta

    def phi_theta_to_equi_xy(self):
        self.x_equi = (self.phi / 360 + 0.5) * (self.W_equi - 1)
        self.y_equi = (self.theta / 180 + 0.5) * (self.H_equi - 1)
        self.x_equi = self.x_equi.astype(np.float32)
        self.y_equi = self.y_equi.astype(np.float32)

    def draw_center(self):
        center_x, center_y = self.equi_img.shape[1] // 2, self.equi_img.shape[0] // 2
        radius = 50
        cv2.circle(self.equi_img, (center_x, center_y), radius, (0, 0, 255), thickness=cv2.FILLED)
        cv2.imwrite("cercle drawn.png", self.equi_img)

    def remap_equi_xy_to_mn(self):
        self.viewport = cv2.remap(self.equi_img,
                                  self.x_equi,
                                  self.y_equi,
                                  cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_WRAP)

    def project_to_viewport(self):
        self.y_equi_int = np.int32(np.round(self.y_equi))
        self.x_equi_int = np.int32(np.round(self.x_equi))

        self.y_equi_int = np.clip(self.y_equi_int, 0, self.equi_img.shape[0] - 1)

        self.x_equi_int = np.clip(self.x_equi_int, 0, self.equi_img.shape[1] - 1)

    def add_centered_text(self, vp_text, viewport):
        # Get image dimensions
        height, width, _ = viewport.shape

        # Define the font, scale, and color
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 9   # Adjust the scale as needed
        font_color = (0, 255, 0)  # Red color in BGR
        font_thickness = 9

        # Get the size of the text
        (text_width, text_height), baseline = cv2.getTextSize(vp_text, font, font_scale, font_thickness)

        # Calculate the center position
        x = (width - text_width) // 2
        y = (height + text_height) // 2

        # Put the text on the image
        cv2.putText(viewport, vp_text, (x, y), font, font_scale, font_color, font_thickness, lineType=cv2.LINE_AA)
        return viewport

    def cubic_interpolation_vectorized(self, img, x, y):
        """
        Performs cubic interpolation for pixels in the image using vectorized operations.

        Args:
            img (numpy.ndarray): The input image.
            x (numpy.ndarray): The x-coordinates of the pixels.
            y (numpy.ndarray): The y-coordinates of the pixels.

        Returns:
            numpy.ndarray: The interpolated pixel values.
        """
        x = np.clip(x, 0, img.shape[1] - 1)
        y = np.clip(y, 0, img.shape[0] - 1)

        x0 = np.floor(x).astype(int)
        x1 = x0 + 1
        y0 = np.floor(y).astype(int)
        y1 = y0 + 1

        x1 = np.clip(x1, 0, img.shape[1] - 1)
        y1 = np.clip(y1, 0, img.shape[0] - 1)

        pixels = [
            img[y0, x0], img[y0, x1],
            img[y1, x0], img[y1, x1]
        ]

        alphas = [
            (x1 - x) * (y1 - y),
            (x - x0) * (y1 - y),
            (x1 - x) * (y - y0),
            (x - x0) * (y - y0)
        ]

        interpolated_values = np.sum([p * a[:, :, np.newaxis] for p, a in zip(pixels, alphas)], axis=0)

        return np.uint8(interpolated_values)

    def remap_viewport_to_equirectangular(self, viewport):  # viewport, x, y, equirect
        """
        Remaps a viewport image back to an equirectangular image.

        Args:
            viewport (numpy.ndarray): The viewport image.
            x (numpy.ndarray): The x-coordinates used for the original mapping.
            y (numpy.ndarray): The y-coordinates used for the original mapping.
            equirect_shape (tuple): The shape of the target equirectangular image (height, width, channels).

        Returns:
            numpy.ndarray: The remapped equirectangular image.
        """
        # Create an empty equirectangular image
        # Round x and y to nearest integer and clip to image boundaries
        x = self.x_equi
        y = self.y_equi
        equirect = self.equi_img
        equirect_shape = equirect.shape
        x_mapped = np.clip(np.round(x).astype(int), 0, equirect_shape[1] - 1)
        y_mapped = np.clip(np.round(y).astype(int), 0, equirect_shape[0] - 1)

        # Create a mask for valid coordinates
        valid_mask = (x_mapped >= 0) & (x_mapped < equirect_shape[1]) & \
                     (y_mapped >= 0) & (y_mapped < equirect_shape[0])

        # Remap the viewport pixels to the equirectangular image
        equirect[y_mapped[valid_mask], x_mapped[valid_mask]] = viewport[valid_mask]

        return equirect







