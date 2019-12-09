# Python program for Detection of a 
# specific color(blue here) using OpenCV with Python
from backupPyimageSearch import *
from usefullFunctions import *

def preprocess_2(image):
    # %%
    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    showImage(gray,"grey")
    # %% Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (13, 13), 0)
    showImage(blurred,"blurred")

    # %% Apply Canny edge detection
    edged = cv2.Canny(blurred, 100, 200)
    showImage(edged,"edged")
    # skeleton=skeletonize(edged)
    # showImage(skeleton,"skelotten")

    # dilate = cv2.dilate(edged, None, iterations=4)
    # showImage(dilate,"dilate")
    # #
    # erode = cv2.erode(dilate, None, iterations=4)
    # showImage(erode,"Erode")
    #
    # mask2 = np.ones(image.shape[:2], dtype="uint8") * 255
    # cnts = cv2.findContours(erode.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    # newimage = cv2.bitwise_and(erode.copy(), dilate.copy(), mask=mask2)
    # showImage(newimage,"bitwise_and")

    # %%
    # find contours in the edge map, then sort them by their size in descending order
    cnts = findSortedContours(edged)
    # finds contours with 4 vertices
    displayCnt = findFirstFourSidedContour(cnts)
    # extract the display, apply a perspective transform to it
    warped = four_point_transform(gray, displayCnt.reshape(4, 2))
    showImage(warped,"warped begining")

    #thresh=adjustmentThreasholding(warped)

    # %%
    #threshold the warped image and cleanup using morphological operations
    thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_OTSU)[1]
    showImage(thresh,"threshold")
    #
    # warped=applyMorphologicalOperations(thresh)
    # showImage(warped,"morphology")
    warped=thresh

    # resize region of interest
    resized = warped[int(warped.shape[1] / 8): int(warped.shape[1] / -8), int(warped.shape[1] / 2): -1]
    showImage(resized,"resized")
    return resized

if __name__ == '__main__':
    # processImage()
    path="Frame.png"
    path="example.jpg"
    #path="Digital_Meter_1.jpg"
    image=cv2.imread(path)
    image = imutils.resize(image, height=400, width=400)
    dst=preprocess_2(image)
    showImage(dst,"INPUT_FOR_OCR.jpg")
    output=getOcr(dst)
    putText(image,output)