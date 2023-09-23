from rectification import rectify_image
from object_measurement import get_reference_object,select_box_roi,calculate_dimen

from skimage import feature, color, transform, io
import numpy as np
import cv2

import time

if __name__ == '__main__':
    import sys
    image_name = sys.argv[-1]
    # image = io.imread(image_name)
    # image = Image.fromarray(img_as_ubyte(image))

    print("Rectifying {}".format(image_name))
    save_name = '.'.join(image_name.split('.')[:-1]) + '_warped.png'
    warped_image = rectify_image(image_name, 4, algorithm='independent')
    # print(warped_image)
    # Image.fromarray(warped_image.astype(np.uint8)).save("test_image.png")
    io.imsave(save_name, (warped_image* 255).astype(np.uint8))

    time.sleep(2)

    width=None
    pixelPerMetric = None
    image_with_refer=None

    image = cv2.imread(save_name)

    if not width:
        pixelPerMetric,width,image_with_refer = get_reference_object(image,pixelPerMetric)

    


    roi_2 =select_box_roi("Select_Measuring_Object",image_with_refer)
    calculate_dimen(roi_2,image_with_refer,pixelPerMetric,width)