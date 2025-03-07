import cv2 as cv
import os
import json
from pathlib import Path
from typing import List
from ultralytics import YOLO # type: ignore

from abc import ABC, abstractmethod

class ImageProcessor(ABC):

    @abstractmethod
    def execute(self, *args, **kwargs):
        pass


class FaceDetector(ImageProcessor):

    def execute(self, frame:cv.typing.MatLike, face_classifier:cv.CascadeClassifier):
        gray_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
        
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
        
        return frame


class InteligenceDetector(ImageProcessor):

    def __init__(self) -> None:
        self.model = "yolo11n-seg.pt"

    def execute(self, frame:cv.typing.MatLike, *args, **kwargs) -> cv.typing.MatLike:
        print("Modelo selecionado:: ", self.model)
        _model = YOLO(self.model) # carregando modelo treinado
        
        
        results = _model.predict(frame, save=False, conf=0.6, device="1")

        for item in results:
            object_detected = item.boxes
            for data in object_detected:
                #print(data)
                x, y, w, h = data.xyxy[0]
                x, y, w, h = int(x), int(y), int(w), int(h)
                label_cls_id = int(data.cls[0])
                confiance = float(data.conf[0])
                label = _model.names[label_cls_id]
                
                print("Detectado Objeto:: ", label, " Confiança:: ", confiance)
                
                if confiance > 0.8:
                    cv.putText(frame, f"{label} - {'%.2f' % confiance}%", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2) # type: ignore
                    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

        return frame

class StreamingProcess():

    def __init__(self, instance_process:ImageProcessor):
        self.instance_process = instance_process

    def execute(self, save_frames:bool = False):
        print("Hello from face-detection!")
        __count = 0
        success = True
        cam = cv.VideoCapture(0)
        
        while success:
            success, frame = cam.read()
            
            if save_frames:
                name_image = os.path.join("images", "frame%d.jpg"%__count)
                cv.imwrite(name_image, frame) 
                __count += 1
            
            if not success:
                break
 
            face_classifier = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml") # type: ignore
            
            frame = cv.flip(frame, 1)
            frame = self.instance_process.execute(frame, face_classifier)
         
            
            cv.imshow("Face Detection", frame)
            
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            
        cam.release()
        cv.destroyAllWindows()


class FilesProcessor():

    def __init__(self, path_images:str, instance_process:ImageProcessor):
        self.path_images = path_images
        self.instance_process = instance_process

    def execute(self, save_frames:bool = False):
        for file in Path(self.path_images).glob("*.jpg"):
            image_name = str(file.relative_to("."))
            image = cv.imread(image_name)
            image = self.instance_process.execute(image)
            
            if save_frames:
                os.makedirs("result", exist_ok=True)
                image_result = os.path.join("result" , image_name)
                cv.imwrite(image_result, image) # type: ignore



def main():
    process = StreamingProcess(InteligenceDetector())
    #process = FilesProcessor(os.path.join("images"), InteligenceDetector())
    process.execute(False)

if __name__ == "__main__":
    main()
