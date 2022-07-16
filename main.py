import sys
import cv2 as cv
import numpy as np


def main(argv):
    default_file = 'smarties.png'
    filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)
    # Check if image is loaded fine
    if src is None:
        print('Error opening image!')
        print('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
        return -1

    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    gray = cv.medianBlur(gray, 5)

    rows = gray.shape[0]
    circles = cv.HoughCircles(
        gray,  # 灰度
        cv.HOUGH_GRADIENT,  # OpenCV
        1,  # 分辨率的反比
        rows / 8,  # 行
        param1=100,  # 內部邊緣監測的上限值
        param2=30,  # 中心檢測值
        minRadius=1,  # 最小半徑 的判斷
        maxRadius=100  # 最大半徑 的判斷
    )

    # 繪製的圓形如果存在(成功繪製)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])

            # circle center 圓心
            cv.circle(src,
                      center,  # 圓心位置
                      1,  # 繪製圓形的半徑
                      (0, 100, 100),  # 線條顏色
                      3  # 線條寬度
            )

            # circle outline
            radius = i[2]
            cv.circle(src,
                      center,  # 圓心位置
                      radius,  # 半徑長度
                      (255, 0, 255),  # 線條顏色
                      3  # 線條寬度
            )

    cv.imshow("detected circles", src)
    cv.waitKey(0)

    return 0


if __name__ == "__main__":
    main(sys.argv[1:])