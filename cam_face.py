# import the opencv library
import cv2
import face_recognition

knownEncodings = []
knownNames = "Gurkirat"

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
image = cv2.imread('gurkirat.jpg', 0)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
boxes = face_recognition.face_locations(rgb, model='hog')
encodings = face_recognition.face_encodings(rgb, boxes)

for encoding in encodings:
    knownEncodings.append(encoding)
    # knownNames.append(name)


# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    frame = cv2.flip(frame, 1)
    # Display the resulting frame

    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(frame, 1.1, 5)
    encodings_vid = face_recognition.face_encodings(frame)
    name = "Unknown"
    for encoding_vid in encodings_vid:
        matches = face_recognition.compare_faces(knownEncodings, encoding_vid)
        if True in matches:
            name = knownNames

    for(x, y, w, h) in faces:
        cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imshow('frame', frame)


      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()