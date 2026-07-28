import cv2
import numpy as np



class ColorDetector:


    def __init__(self):

       self.pixel_offset=np.array([0,28])


    def detect_red(
        self,
        rgb
    ):

        """
        输入RGB
        输出红色物体pixel中心
        """

        hsv = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2HSV
        )

        # red两个范围

        lower1 = np.array(
            [0,100,100]
        )

        upper1 = np.array(
            [10,255,255]
        )

        lower2 = np.array(
            [170,100,100]
        )

        upper2 = np.array(
            [180,255,255]
        )

        mask1 = cv2.inRange(
            hsv,
            lower1,
            upper1
        )

        mask2 = cv2.inRange(
            hsv,
            lower2,
            upper2
        )


        mask = mask1 | mask2

        contours,_ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours)==0:
            return None

        # 最大红色区域

        c = max(
            contours,
            key=cv2.contourArea
        )

        M=cv2.moments(c)

        if M["m00"]==0:
            return None

        u=int(
            M["m10"]/M["m00"]
        )

        v=int(
            M["m01"]/M["m00"]
        )

        return np.array([u,v]) 