from darkflow.net.build import TFNet
import cv2
import time

# options = {"model": "cfg/yolov2.cfg", "load": "yolov2/yolov2.weights", "threshold": 0.1}
options = options = {"model": "cfg/yolov2.cfg", "load": "yolov2/yolov2.weights", "threshold": 0.7, "gpu": 0.3}
tfnet = TFNet(options)


cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # tfnet.

    # Display the resulting frame
    result = tfnet.return_predict(frame)
    print(result)

    for dictionary in result:
        label = dictionary["label"]
        top_left = dictionary["topleft"]
        bottom_right = dictionary["bottomright"]
        cv2.rectangle(frame, (top_left['x'], top_left['y']), (bottom_right['x'], bottom_right['y']), (0, 255, 0), 3)

        cv2.putText(frame, label, (top_left['x'], top_left['y']), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


# imgcv = cv2.imread("./sample_img/sample_dog.jpg")
# # print("predicting")
# # t0 = time.time()
# # result = tfnet.return_predict(imgcv)
# # t1 = time.time()
# # total = t1-t0
# # print(total)
# # print(result)