import cv2
import numpy as np
import PIL
from PIL import Image
from usefullFunctions import *
from skimage import exposure
import imutils
import matplotlib.pyplot as plt
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

img = cv2.imread("Digital_Meter_1.jpg")
showImage(img,"original")
## convert to hsv
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


mask = cv2.inRange(hsv, (0, 25, 25), (70, 255,255))

## slice the green
imask = mask>0
green = np.zeros_like(img, np.uint8)
green[imask] = img[imask]

## save
# cv2.imwrite("green.png", green)
showImage(green,"greening")


gray = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)
showImage(gray,"graying")
# print(pytesseract.image_to_string(gray))
print(pytesseract.image_to_string(gray, lang="letsgodigital"))


# im_gray = cv2.imread("./WhatsApp Image 2019-12-03 at 9.28.39 AM.jpeg", cv2.IMREAD_GRAYSCALE)
(thresh, im_bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
showImage(im_bw,"thresholding")

im_ = cv2.subtract(255, im_bw)
showImage(im_,"substract")

print(pytesseract.image_to_string(im_))

print(pytesseract.image_to_string(im_bw,  lang="letsgodigital"))



gray_ = cv2.subtract(255, im_)
print(pytesseract.image_to_string(gray_, lang="letsgodigital"))

showImage(gray_,"graying_threshold")



# Blur the image
res = cv2.GaussianBlur(gray_,(13,13), 0)
showImage(res,"gausian_blur")

print(pytesseract.image_to_string(res, lang="letsgodigital"))
# pytesseract.image_to_string(res)


# Edge detection
edged = cv2.Canny(res, 100, 190)
showImage(edged,"edged")


# Dilate it , number of iterations will depend on the image
dilate = cv2.dilate(edged, None, iterations=0)
showImage(dilate,"dilated")


# perform erosion
erode = cv2.erode(dilate, None, iterations=0)
showImage(erode,"eroded")




# make an empty mask
mask2 = np.ones(erode.shape[:2], dtype="uint8") * 255


 # Remove ignored contours
newimage = cv2.bitwise_and(erode.copy(), dilate.copy(), mask=mask2)
showImage(newimage,"ignore_contour")


# Again perform dilation and erosion
newimage = cv2.dilate(newimage,None, iterations=3)
newimage = cv2.erode(newimage,None, iterations=0)
showImage(newimage,"dilation_erosion")


ret,newimage = cv2.threshold(newimage,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
showImage(newimage,"threshold")
# Tesseract OCR
print( pytesseract.image_to_string(newimage))
showImage(newimage)

print(pytesseract.image_to_string(newimage))

grayImage = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)
showImage(grayImage)

(thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
showImage(blackAndWhiteImage)


print(pytesseract.image_to_string(blackAndWhiteImage))


image_file = Image.open("./example.jpg") # open colour image
image_file= image_file.convert('L') # convert image to monochrome - this works
showImage(blackAndWhiteImage)


image_file= image_file.convert('1')
showImage(blackAndWhiteImage)
