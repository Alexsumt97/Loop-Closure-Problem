import cv2
import numpy as np
import re

maindir = "101_ObjectCategories"
# fh = open("new2.txt")
count=0
namespace = ""

img1=cv2.imread("image_0001.jpg",0)

matched_img = None
max_match = 0
max_mr = None
img2= None
ant_match =0
fh = open("new2.txt")
for line in fh:
    line = line.rstrip()
    if re.search(".jpg",line):
        img2path = namespace+line
        
        img2=cv2.imread(img2path,0)
        print(count,"Matching with",img2path)
        orb=cv2.ORB_create()
        kp1,des1=orb.detectAndCompute(img1,None)
        kp2,des2=orb.detectAndCompute(img2,None)

        bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
        matches=bf.match(des1,des2)
        m_r=cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=2)

     
        print("Matched value",len(matches))
        if len(matches) > max_match:
            max_match = len(matches)
            matched_img = img2
            max_mr = m_r
        print("Maximum upto now:",max_match)
        count=count+1
        
    else:
        namespace = maindir+"\\"+line[:-2] +"\\"

print("A Match Is Found")
print(max_match)
cv2.imshow('Database_Image',img1)
cv2.imshow('Matched_Image', matched_img)
cv2.imshow('Keypoints',max_mr)
cv2.waitKey(0)
cv2.destroyAllWindows()
