import cv2 as cv
hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
cam = cv.VideoCapture('video/people.mp4')
while True:
    ret, frame = cam.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    if ret:
        boxes, weights = hog.detectMultiScale(gray,
                                              winStride=(4, 4),
                                              padding=(4, 4),
                                              scale=1.05) #Tunning Parameters
        for (x, y, w, h) in boxes:
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.imshow('Pedestrians Detection (Live)', frame)
        key = cv.waitKey(1)
        if(key==81 or key==113):
            break
    else:
        break
cam.release()
cv.destroyAllWindows()