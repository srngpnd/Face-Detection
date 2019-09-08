# Import opencv library
import cv2

# Since we need to track face, we will use the frontal face xml
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Start video capture
cap = cv2.VideoCapture(0) 

# Initiate control flow
while 1:  
  
    # Start reading image
    ret, img = cap.read()  
  
    # Pass video capture into openCV
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    # Detect faces  
    faces = face.detectMultiScale(gray, 1.3, 5) 
    
    # Highlight the faces using rectangles
    for (x,y,w,h) in faces: 
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)  
        roi_gray = gray[y:y+h, x:x+w] 
        roi_color = img[y:y+h, x:x+w] 
    
    # Display video
    cv2.imshow('img',img) 
    
    # Terminate the control flow
    k = cv2.waitKey(30) & 0xff
    if k == 27: 
        break

# release video capture  
cap.release() 
  
# close the window
cv2.destroyAllWindows()  