import cv2

cap = cv2.VideoCapture("test1.mp4")
i=1

while(cap.isOpened()):
    frameId = cap.get(1) #current frame number
    ret, frame = cap.read()
    if (frameId % 10 == 0):
        filename = "input_images/%i.jpg"%i
        cv2.imwrite(filename, frame)
        i += 1
cap.release()
