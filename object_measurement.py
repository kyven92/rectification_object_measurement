from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import argparse
import numpy as np
import imutils
import cv2
import time



def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def show_image(title, image, destroy_all=True):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    
    # Using resizeWindow()
    cv2.resizeWindow(title, 1920, 1080)

    cv2.imshow(title, image)
    cv2.waitKey(0)
    if destroy_all:
        cv2.destroyAllWindows()


def select_box_roi(title,img):

    # Get the mouse coordinates
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    
    # Using resizeWindow()
    cv2.resizeWindow(title, 1920, 1080)
    r = cv2.selectROI(title,img)

    # Create a region of interest
    cropped_image = img[int(r[1]):int(r[1]+r[3]), 
                      int(r[0]):int(r[0]+r[2])]

    show_image("ROI",cropped_image,False)
    return r

def find_contours(roi,image):

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    edged = cv2.Canny(gray, 50, 100)
    # show_image("Edged", edged, True)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    show_image("erode and dilate", edged, True)

    cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    print("Total number of contours are: ", len(cnts))


    for c in cnts:
        if cv2.contourArea(c) < 1000:
            continue


        orig = roi.copy()
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

        show_image("Contoured",orig,True)

def calculate_dimen(roi_coord,image,pixelPerMetric,width=None):

    print(f"###### ROI: {roi_coord}")
    start=(int(roi_coord[0]),int(roi_coord[1]))
    end=(int(roi_coord[0]+roi_coord[2]),int(roi_coord[1]+roi_coord[3]))

    tl = (int(roi_coord[0]),int(roi_coord[1]))
    tr = (int(roi_coord[0]+roi_coord[2]),int(roi_coord[1]))
    br = (int(roi_coord[0]+roi_coord[2]),int(roi_coord[1]+roi_coord[3]))
    bl = (int(roi_coord[0]),int(roi_coord[1]+roi_coord[3]))

    im_arr_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.rectangle(im_arr_bgr, start,end, color=(0, 255, 0), thickness=2)

    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    cv2.circle(im_arr_bgr, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(im_arr_bgr, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(im_arr_bgr, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(im_arr_bgr, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

    cv2.line(im_arr_bgr, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
    cv2.line(im_arr_bgr, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    if not width:
        width = float(input("Please provide width of ROI in inches: "))


    if pixelPerMetric is None:
        pixelPerMetric = dB / width

    dimA = dA / pixelPerMetric
    dimB = dB / pixelPerMetric

    cv2.putText(im_arr_bgr, "{:.1f}in".format(dimA), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(im_arr_bgr, "{:.1f}in".format(dimB), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


    show_image("Contoured",im_arr_bgr,True)

    return pixelPerMetric,width,im_arr_bgr


def get_reference_object(image,pixelPerMetric):

    roi =select_box_roi("Get_Reference_Object",image)
    pixelPerMetric,width,image_with_refer = calculate_dimen(roi,image,pixelPerMetric)

    return pixelPerMetric,width,image_with_refer









if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to the input image")
    # ap.add_argument("-w", "--width", type=float, required=True, help="width of the left-most object in the image (in inches)")
    args = vars(ap.parse_args())

    width=None
    pixelPerMetric = None
    image_with_refer=None

    image = cv2.imread(args["image"])

    if not width:
        pixelPerMetric,width,image_with_refer = get_reference_object(image,pixelPerMetric)

    


    roi_2 =select_box_roi("Select_Measuring_Object",image_with_refer)
    calculate_dimen(roi_2,image_with_refer,pixelPerMetric,width)

    time.sleep(2)