from random import randint

from darkflow.net.build import TFNet
import cv2

# options = {"model": "cfg/yolov2.cfg", "load": "yolov2/yolov2.weights", "threshold": 0.1}
colors = []
for i in range(16):
    colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))

options = options = {"model": "cfg/yolov2.cfg", "load": "yolov2/yolov2.weights", "threshold": 0.5, "gpu": 0.3}
tfnet = TFNet(options)
cap = cv2.VideoCapture(0)

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    # Display the resulting frame
    result = tfnet.return_predict(frame)
    print(result)
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    for dictionary in result:
        label = dictionary["label"]
        top_left = dictionary["topleft"]
        bottom_right = dictionary["bottomright"]
        color = hash(label)%15
        cv2.rectangle(frame, (top_left['x'], top_left['y']), (bottom_right['x'], bottom_right['y']), colors[color], 2)
        cv2.putText(frame, label, (top_left['x'], top_left['y']), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=3)
        cv2.putText(frame, label, (top_left['x'], top_left['y']), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), thickness=1)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
