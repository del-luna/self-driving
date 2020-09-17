import cv2
import lane_detection

if __name__ == "__main__":
    cap = cv2.VideoCapture('./challenge.mp4')

    while cap.isOpened():
        ret, image = cap.read()
        result = lane_detection.run(image)
        cv2.imshow('result', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyWindow()
