import glob
import cv2

fps = 20
images = glob.glob('data/images/*.png')

for img_path in images:
    frame = cv2.imread(img_path)
    cv2.imshow('Attention Estimation', frame)
    cv2.waitKey(1/fps)

