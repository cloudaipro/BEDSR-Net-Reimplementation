import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

demodata_path = './demodata'
demodata = os.listdir(demodata_path)

results_path = './results'

figure = plt.figure(figsize=(9 * 3, 2 * 3 * len(demodata)))
dataSize = len(demodata)
for i, d in enumerate(demodata):
    im_path = os.path.join(demodata_path, d)
    print(im_path)

    img = cv2.imread(im_path, -1)  # cv::IMREAD_UNCHANGED = -1,
    rgb_planes = cv2.split(img)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    print(os.path.join(results_path, 'cv_' + d))
    cv2.imwrite(os.path.join(results_path, 'cv_' + d), result)
    cv2.imwrite(os.path.join(results_path, 'cvn_' + d), result_norm)

    plt.subplot(dataSize, 3, i * 3 + 1)
    plt.title(d + ' input image')
    plt.imshow(img)

    plt.subplot(dataSize, 3, i * 3 + 2)
    plt.title(d + ' shadow removal normalized')
    plt.imshow(result_norm)

    plt.subplot(dataSize, 3, i * 3 + 3)
    plt.title(d + ' shadow removal image')
    plt.imshow(result)

# plt.show()
plt.savefig('./my_plot.png')
