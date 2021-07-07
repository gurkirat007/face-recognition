import face_recognition
import imutils
import pickle
import time
import cv2
import os
 
#preparing cascade for use
cascPathface = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPathface)
# gettng data of known faces
data = pickle.loads(open('face_enc', "rb").read())
# load the image to detect faces in
image = cv2.imread('gk.jpg')


rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#convert image to Greyscale for haarcascade
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(gray,
                                    scaleFactor=1.5,
                                    minNeighbors=10,
                                    minSize=(60, 60),
                                    flags=cv2.CASCADE_SCALE_IMAGE)

# the facial embeddings for face in input
encodings = face_recognition.face_encodings(rgb)
names = []

# we have multiple embeddings for multiple faces
for encoding in encodings:
    # loop through all encoding
    matches = face_recognition.compare_faces(data["encodings"],
    encoding)
    #set name = unknown if no encoding matches
    name = "Unknown"
    if True in matches:
        # Find positions at which we get True and store them
        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
        counts = {}
        # loop over the matched indexes and maintain a count for
        # each recognized face face
        for i in matchedIdxs:
            #Check the names at respective indexes we stored in matchedIdxs
            name = data["names"][i]
            # print(name)
            #increase count for the name we got
            counts[name] = counts.get(name, 0) + 1
            #set name which has highest count
            name = max(counts, key=counts.get)
 
 
        # update the list of names
    names.append(name)
print(names)
names.reverse()
        # print(names)
        # loop over the recognized faces
for ((x, y, w, h), name) in zip(faces, names):
            # rescale the face coordinates
            # draw the predicted face name on the image
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
        0.75, (0, 255, 0), 2)
cv2.imshow("Frame", image)
cv2.waitKey(0)

