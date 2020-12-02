import numpy as np
import cv2
from collections import deque

#default trackbar function
def setValues(x):
   print("")

# Small trackbar which works as Marker Color Picker
cv2.namedWindow("Color detectors")
cv2.createTrackbar("Hue Max", "Color detectors", 179, 180,setValues)
cv2.createTrackbar("Saturation Max", "Color detectors", 255, 255,setValues)
cv2.createTrackbar("Value Max", "Color detectors", 255, 255,setValues)
cv2.createTrackbar("Hue Min", "Color detectors", 81, 180,setValues)
cv2.createTrackbar("Saturation Min", "Color detectors", 88, 255,setValues)
cv2.createTrackbar("Value Min", "Color detectors",99, 255,setValues)

# Each Array Handle different color point.
bluepoints = [deque(maxlen=1024)]
greenpoints = [deque(maxlen=1024)]
redpoints = [deque(maxlen=1024)]
yellowpoints = [deque(maxlen=1024)]

# Each index will be used to mark the points in particular arrays of Different color
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

#The kernel to be used for dilation purpose
#Dilation is done here so that impurities are minimum and we get a clear image of our marker
kernel = np.ones((5,5),np.uint8)

#The format is BGR not RGB in opencv
# blue , green , red , yellow
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0

# Here is code for white Canvas
paintWindow = np.zeros((471,636,3)) + 255
paintWindow = cv2.rectangle(paintWindow, (40,1), (140,65), (0,0,0), 2)
paintWindow = cv2.rectangle(paintWindow, (160,1), (255,65), colors[0], -1)
paintWindow = cv2.rectangle(paintWindow, (275,1), (370,65), colors[1], -1)
paintWindow = cv2.rectangle(paintWindow, (390,1), (485,65), colors[2], -1)
paintWindow = cv2.rectangle(paintWindow, (505,1), (600,65), colors[3], -1)

cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_4)
cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,cv2.LINE_4 )
cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_4)
cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),2, cv2.LINE_4)
cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 2, cv2.LINE_4)
cv2.namedWindow('Canvas', cv2.WINDOW_AUTOSIZE)


# Loading the default webcam.
cap = cv2.VideoCapture(0)

#Always looping because its a video
while True:
    # Reading the frame from the camera
    ret, frame = cap.read()
    #Flipping the frame to see same side of yours
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    u_hue = cv2.getTrackbarPos("Hue Max", "Color detectors")
    u_saturation = cv2.getTrackbarPos("Saturation Max", "Color detectors")
    u_value = cv2.getTrackbarPos("Value Max", "Color detectors")
    l_hue = cv2.getTrackbarPos("Hue Min", "Color detectors")
    l_saturation = cv2.getTrackbarPos("Saturation Min", "Color detectors")
    l_value = cv2.getTrackbarPos("Value Min", "Color detectors")
    Upper_hsv = np.array([u_hue,u_saturation,u_value])
    Lower_hsv = np.array([l_hue,l_saturation,l_value])


    # Adding the colour buttons to the live frame for colour access
    frame = cv2.rectangle(frame, (40,1), (140,65), (122,122,122), -1)
    frame = cv2.rectangle(frame, (160,1), (255,65), colors[0], -1)
    frame = cv2.rectangle(frame, (275,1), (370,65), colors[1], -1)
    frame = cv2.rectangle(frame, (390,1), (485,65), colors[2], -1)
    frame = cv2.rectangle(frame, (505,1), (600,65), colors[3], -1)
    cv2.putText(frame, "CLEAR ALL", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_4)
    cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_4)
    cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_4)
    cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_4)
    cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 2, cv2.LINE_4)


    # Identifying the pointer by making its mask
    Mask = cv2.inRange(hsv, Lower_hsv, Upper_hsv)
    Mask = cv2.erode(Mask, kernel, iterations=1)
    Mask = cv2.morphologyEx(Mask, cv2.MORPH_OPEN, kernel)
    Mask = cv2.dilate(Mask, kernel, iterations=1)

    # Find contours for the pointer after idetifying it
    cnts,_ = cv2.findContours(Mask.copy(), cv2.RETR_EXTERNAL,
    	cv2.CHAIN_APPROX_SIMPLE)
    center = None

    # Ifthe contours are formed
    if len(cnts) > 0:
    	# sorting the contours to find biggest
        cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
        # Get the radius of the enclosing circle around the found contour
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        # Draw the circle around the contour
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        # Calculating the center of the detected contour
        M = cv2.moments(cnt)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        # Now checking if the user wants to click on any button above the screen
        if center[1] <= 65:
            if 40 <= center[0] <= 140: # Clear Button
                bluepoints = [deque(maxlen=512)]
                greenpoints = [deque(maxlen=512)]
                redpoints = [deque(maxlen=512)]
                yellowpoints = [deque(maxlen=512)]

                blue_index = 0
                green_index = 0
                red_index = 0
                yellow_index = 0

                paintWindow[67:,:,:] = 255
            elif 160 <= center[0] <= 255:
                    colorIndex = 0 # Blue
            elif 275 <= center[0] <= 370:
                    colorIndex = 1 # Green
            elif 390 <= center[0] <= 485:
                    colorIndex = 2 # Red
            elif 505 <= center[0] <= 600:
                    colorIndex = 3 # Yellow
        else :
            if colorIndex == 0:
                bluepoints[blue_index].appendleft(center)
            elif colorIndex == 1:
                greenpoints[green_index].appendleft(center)
            elif colorIndex == 2:
                redpoints[red_index].appendleft(center)
            elif colorIndex == 3:
                yellowpoints[yellow_index].appendleft(center)
    # Append the next deques when nothing is detected to avois messing up
    else:
        bluepoints.append(deque(maxlen=512))
        blue_index += 1
        greenpoints.append(deque(maxlen=512))
        green_index += 1
        redpoints.append(deque(maxlen=512))
        red_index += 1
        yellowpoints.append(deque(maxlen=512))
        yellow_index += 1

    # Draw lines of all the colors on the canvas and frame
    points = [bluepoints, greenpoints, redpoints, yellowpoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

    # Show all the windows
    cv2.imshow("Tracking", frame)
    cv2.imshow("Canvas", paintWindow)
    cv2.imshow("mask",Mask)

    # If the 'q' key is pressed then stop the application
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and all resources
cap.release()
cv2.destroyAllWindows()