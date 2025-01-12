import cv2 as cv
import numpy as np


class FaceDector():
    
    def detect_bounding_box(self, frame:cv.typing.MatLike, face_classifier:cv.CascadeClassifier):
        gray_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
        
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
        
        return faces


    def execute(self):
        print("Hello from face-detection!")
        success = True
        cam = cv.VideoCapture(0)    
        
        while success:
            success, frame = cam.read()
            
            if not success:
                break

            
            face_classifier = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
            
            self.detect_bounding_box(frame, face_classifier)
            
            frame = cv.flip(frame, 1)
            cv.imshow("Face Detection", frame)
            
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            
        cam.release()
        cv.destroyAllWindows()


def main():
    face_detector = FaceDector()
    face_detector.execute()

if __name__ == "__main__":
    main()
