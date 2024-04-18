import cv2 as cv
capture = cv.VideoCapture(0)
haar_cascade = cv.CascadeClassifier(r'C:\Users\Parth\Desktop\DataScience\OPENCV\Face_Detection-Open-CV\haar_face.xml')
while True:
    isTrue,frame = capture.read()
    grey = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)#converting the frame to grey
    face_co_ordinaties = haar_cascade.detectMultiScale(grey,scaleFactor=1.1,minNeighbors=2)#this gives us the co-ordinates of the detected face
    #to draw a rectange on the frame
    for (x,y,w,h) in face_co_ordinaties:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),thickness =2)
        #cv.rectange(img,point1 or starting point, daigonal point of the rectange, colors of the line, and thickness of line)
    cv.imshow('Face_detected',frame)#display the frame , now there will be rectange drawn on the image
    if cv.waitKey(20) & 0xFF== ord('q'):
        break

capture.release()
cv.destroyWindow()

