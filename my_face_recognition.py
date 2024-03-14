import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime

# step1 im/port the image
path = "C:\\Users\\KIIT\\Downloads\\photos_face_detection"
image=[]
Student_Name=[]
myList=os.listdir(path)
print(myList)
for cl in myList:
    curImg=cv2.imread(f'{path}/{cl}')
    image.append(curImg)
    Student_Name.append(os.path.splitext(cl)[0])
    

print(Student_Name)

#step2 find encoding in the image

def findEncodings(image):
    encodeList=[]
    for img in image:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode =face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList
def AttendenceMark(name):
    with open(r'C:\Users\KIIT\OneDrive\Documents\VS CODE\opencv\Attendence.csv','+r') as f:
        myDataList=f.readlines()
        nameList=[]
        for line in myDataList:
            entry=line.split(',')

        if name not in nameList:
            now=datetime.now()
            dtString=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
    

encodeListKnown=findEncodings(image)
print("encoding complete")

#step3->find the matches between our encoding

cap=cv2.VideoCapture(0)

while True:
    sucess,img=cap.read()
    imgs=cv2.resize(img,(0,0),None,0.25,0.25)
    imgs=cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgs)
    encodesCurFrame = face_recognition.face_encodings(imgs, facesCurFrame)
    

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
         
         matches=face_recognition.face_distance(encodeListKnown,encodeFace)
         faceDis=face_recognition.face_distance(encodeListKnown,encodeFace)
         print(faceDis)
         matchIndex=np.argmin(faceDis)
         if matches[matchIndex]:
             name=Student_Name[matchIndex].upper()
             print(name)
             y1,x2,y2,x1=faceLoc
             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # No error in this line
             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), cv2.FILLED)
             
             cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

             AttendenceMark(name)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # No error in this line
        break


    cv2.imshow('webcam',img)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()