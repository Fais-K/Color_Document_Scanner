from skimage.filters import unsharp_mask
import argparse
import numpy as np 
from imutils.perspective import order_points
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

def corner_transform(image, corners):
	rect = order_points(corners)
	(tl, tr, br, bl) = rect

	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	return warped

def set_colortone(img, value=30):
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)

    lim = 255 - value
    v[v > lim] = 255
    value += 15
    v[v == value] = 0

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
edged = cv2.Canny(blur_image, 60, 100)
edged = cv2.dilate(edged, None, iterations = 1)

cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

for c in cnts:
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, peri * 0.04, True)

	if len(approx)==4:
		paper_pnts = approx
		break

cv2.drawContours(image, [paper_pnts], -1, (0,255,0), 2)
cv2.imshow("detected", image)
cv2.waitKey(0)

warped = corner_transform(image, paper_pnts.reshape(4, 2))
# warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
# T = threshold_local(warped, 21, offset = 15, method = "gaussian")
# warped = (warped > T).astype("uint8") * 255
toned_image = set_colortone(warped, 30)

# kernel = np.array([[-1,-1,-1], 
#                    [-1, 9,-1],
#                    [-1,-1,-1]])

# final_image = cv2.filter2D(toned_image, -1, kernel)

final_image = unsharp_mask(toned_image, radius=8, amount=1, multichannel=True)

cv2.imshow("final", final_image)
cv2.waitKey(0)