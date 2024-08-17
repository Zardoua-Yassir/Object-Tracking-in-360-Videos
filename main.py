import cv2
from ViewPortRenderer import ViewPortRenderer


def empty(x):
    pass


if __name__ == '__main__':

    WindowTitle = "Image Viewer"
    cv2.namedWindow(WindowTitle)

    cv2.createTrackbar("In/Out", WindowTitle, 90, 175, empty)
    cv2.createTrackbar("Down/Up", WindowTitle, 90, 180, empty)
    cv2.createTrackbar("Left/Right", WindowTitle, 180, 360, empty)

    # img = cv2.imread("manipulation_data/equi_frame.jpg")
    img = cv2.imread("equi_img_meta.png")

    renderer = ViewPortRenderer(equi_img=img, Wvp=640, Hvp=480)
    ptheta = -1
    pphi = -1
    pFOV = -1

    while True:
        FOV = cv2.getTrackbarPos("In/Out", WindowTitle)
        theta = cv2.getTrackbarPos("Down/Up", WindowTitle) - 90
        phi = cv2.getTrackbarPos("Left/Right", WindowTitle) - 180
        if theta != ptheta or phi != pphi or FOV != pFOV:
            print("theta = ", theta)
            print("phi = ", phi)
            ptheta = theta
            pphi = phi
            pFOV = FOV
            viewport = renderer.render_viewport(fov=FOV, theta_c=theta, phi_c=phi)
            cv2.imshow(WindowTitle, viewport)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break