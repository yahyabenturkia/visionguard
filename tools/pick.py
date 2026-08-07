import cv2, sys
def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"x={x}, y={y}")
        cv2.destroyAllWindows()
img = cv2.imread(sys.argv[1])
img_small = cv2.resize(img, (800, 600))
cv2.namedWindow("Pick ROI")
cv2.setMouseCallback("Pick ROI", click)
cv2.imshow("Pick ROI", img_small)
cv2.waitKey(0)