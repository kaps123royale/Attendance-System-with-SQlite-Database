import cv2
import numpy as np
import sqlite3
import xlwrite,firebase.firebase_ini as fire;
import time
import sys
start=time.time()
period=8

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('recognizer/trainner.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
id=0
filename='filename';
dict = {
            'item1': 1
}

def getProfile(id):
    conn=sqlite3.connect("FaceBase.db")
    cmd="SELECT * FROM People WHERE ID="+str(id)
    cursor=conn.execute(cmd)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile

cam = cv2.VideoCapture(0)
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (255, 255, 255)
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5);
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
        id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        profile=getProfile(id)
        if(profile!=None):
            if(id==1):
                if((str(id)) not in dict):
                    filename=xlwrite.output('attendance','class1',1,str(profile[1]),'yes');
                dict[str(id)]=str(id);
            elif(id==2):
                if ((str(id)) not in dict):
                    filename=xlwrite.output('attendance', 'class1', 2, str(profile[1]), 'yes');
                dict[str(id)]=str(id);
            """elif(str(profile[0])==3):
                if ((str(id)) not in dict):
                    filename =xlwrite.output('attendance', 'class1', 3, str(profile[1]), 'yes');
                    dict[str(id)] = str(id);
                
        else:
            id="Unknown"""
        cv2.putText(im,str(profile[1]),(x,y+h),fontface,fontscale,fontcolor);
    cv2.imshow('im',im)
    if time.time()>start+period:
        break;
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
