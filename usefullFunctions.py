import cv2
import numpy as np
import imutils
import pytesseract
import os
dirname = 'test'

def adjustmentThreasholding(img):
    showImage(img)
    thresh1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 199, 5)

    thresh2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 199, 5)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
    return thresh2

def skeletonize(img):
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    ret, img = cv2.threshold(img, 127, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    while (not done):
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True
    return skel
def applyMorphologicalOperations(thresh):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

def findSortedContours(edged):
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    return sorted(cnts, key=cv2.contourArea, reverse=True)

def findFirstFourSidedContour(cnts):
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if the contour has four vertices, then we have found
        # the display
        if len(approx) == 4:
            return approx

def putText(image,text):
    cv2.putText(image, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 2)
    cv2.imwrite('FINAL.jpg', image)


def showImage(img,default_text="temp",save=False):
    cv2.imshow(default_text, img)
    if(default_text!="temp" and save):
        cv2.imwrite(os.path.join(dirname, default_text+".jpg"), img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def getOcr(image):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    temp = pytesseract.image_to_string(image,lang="letsgodigital", config="whitelist=.0123456789")
    new = ""
    for i in temp:
        new = new + str(i)
        print (i)
    return new