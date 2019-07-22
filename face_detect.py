'''
OpenCV Script -

We will detect a face in this script using Haarcascade algorithm

Process :-


1. Creating a cascade classifier. It will contain the face features.
2. Search for the row and column values of the face in the numpy array (The face rectangle coordinates).
3. Display the image with rectangle box around it.


'''

import cv2

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #creating XML file for face features


img = cv2.imread('mila_kunis.jpg') # change photo path here
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converting the image into grayscale image

faces=face_cascade.detectMultiScale(gray_img, scaleFactor = 1.05, minNeighbors=5) #search the coordinates of the image

'''
detectMultiScale - method for the face rectangle coordinates
scaleFactor - decreases the shape value by 5%, until the face is found. Smaller this value, greater the accuracy
'''

for x, y, w, h in faces:
    img=cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 3)
    

#resize_img = img.resize(img, (int(img.shape[1]/7), int(img.shape[0]/7)))

cv2.imshow('image_with_detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows() # closes the window based on waitKey parameter

print(type(faces))
print(faces)
