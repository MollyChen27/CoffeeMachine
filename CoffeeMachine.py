import cv2
import mediapipe as mp
import os
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
# 640x480是一個常見的解析度
cap.set(3,640) # width 
cap.set(4,480) # height

imgBackground =cv2.imread("Resources/Background.png")
# print(imgBackground.shape) # 背景圖 height, width, channels = (720, 1280, 3)

# importing all the mode images to a list
ModePath = "Resources/Modes"
ModeImgsPath = os.listdir(ModePath) # Modes 中，圖片的檔名
listImgModes = [] # 存放圖片array值
for img_path in ModeImgsPath:
    listImgModes.append(cv2.imread(os.path.join(ModePath, img_path)))
# print(listImgModes[0].shape) # height, width, channels = (720, 433, 3)

# importing all the icons to a list
IconsPath = "Resources/Icons"
IconsImgsPath = os.listdir(IconsPath) # Modes 中，圖片的檔名
listIcons = [] # 存放icon array值
for icon_path in IconsImgsPath:
    listIcons.append(cv2.imread(os.path.join(IconsPath, icon_path)))
print(listIcons[0].shape) # 影片畫面預設 height, width, channels = (720, 1280, 3)

modeType = 0
counter = 0 
selection = -1
selectionList = [-1,-1,-1]
selectionSpeed = 7
modeCenterPosition = [(1136,196),(1000,384), (1136,581)]

detector = HandDetector(maxHands=1, detectionCon=0.5, minTrackCon=0.8)


while True:
    success, img = cap.read()

    hands, img = detector.findHands(img, draw=True, flipType=True)
    
    # overlaying the webcam feed on the background image 
    imgBackground[140:140+480, 50:50+640] = img # 裁剪（截取）圖像的一個區域，把影像塞入圖片中，height 480 ；width 640
    imgBackground[0:720, 847:1280] = listImgModes[modeType]
    
    if hands and modeType < 3:
        # Information for the first hand detected
        hand1 = hands[0]  # Get the first hand detected
        # Count the number of fingers up for the first hand
        fingers1 = detector.fingersUp(hand1)

        # countOfFingersUp = fingers1.count(1)
        if fingers1 == [0,1,0,0,0]:
            if selection !=1:
                counter = 1
            selection = 1
        elif fingers1 == [0,1,1,0,0]:
            if selection !=2:
                counter = 1
            selection = 2
        elif fingers1 == [0,1,1,1,0] or fingers1 == [0,0,1,1,1]:
            if selection !=3:
                counter = 1
            selection = 3
        else:
            selection = -1
            counter = 0 

        if counter > 0 :
            counter+=1
            cv2.ellipse(imgBackground, modeCenterPosition[selection -1], (103,103), 0,0,counter*selectionSpeed,  (0,255,0), 20) #counter*selectionSpeed 代表angle

        if counter*selectionSpeed > 360:
            selectionList[modeType] = selection
            modeType += 1 # 更換背景
            selection = -1 # 設回初始值
            counter = 0 # 設回初始值
        
        if selectionList[0] != -1:
            imgBackground[636:636 + 65, 133:133 + 65] = listIcons[selectionList[0]-1]
        if selectionList[1] != -1:
            imgBackground[636:636 + 65, 340:340 + 65] = listIcons[selectionList[1]+2]
        if selectionList[2] != -1:
            imgBackground[636:636 + 65, 542:542 + 65] = listIcons[selectionList[2]+5]




    cv2.imshow("BackgroundImage", imgBackground)


    key = cv2.waitKey(1)  # Wait for a key event for 1 millisecond
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()