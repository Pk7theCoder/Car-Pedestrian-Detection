import cv2  #importing Computer Vision Library
video = str(input("Enter video file name: "))
#Load pre-trained data
cars_data = cv2.CascadeClassifier("cars.xml")
human_data = cv2.CascadeClassifier("haarcascade_fullbody.xml")
cam = cv2.VideoCapture(video)

while True:
    succ_frame_read, frame = cam.read()
    gs_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    car_cords = cars_data.detectMultiScale(gs_img)  # detects the coordinates of the car
    body_cords = human_data.detectMultiScale(gs_img)
    # Drawing the rectangle over detected area
    for (x, y, w, h) in car_cords:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255),2)
    for (x, y, w, h) in body_cords:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,255),2)
    cv2.imshow("Car and Pedestrian Detector", frame)  # Show image
    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break
cam.release()