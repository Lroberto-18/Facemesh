import cv2
import numpy as np
import mediapipe as mp
import random


class face_mesh_detection():
    def __init__(self, static_image_mode=False,
               max_num_faces=1,
               refine_landmarks=False,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):
               
        self.mode = static_image_mode
        self.max_faces = max_num_faces
        self.refine = refine_landmarks
        self.min_detection = min_detection_confidence
        self.min_tracking = min_tracking_confidence
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(self.mode, self.max_faces,
                                                    self.refine, self.min_detection, self.min_tracking)
        self.mp_draw = mp.solutions.drawing_utils

    def create_mesh(self, image, draw=True):
        self.image_RGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        self.results = self.face_mesh.process(self.image_RGB)
        if draw:
            try:
                for self.face_landmarks in self.results.multi_face_landmarks:
                    self.mp_draw.draw_landmarks(self.image_RGB, self.face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS, 
                                                landmark_drawing_spec = self.mp_draw.DrawingSpec(color=(20,50,50), thickness=1,circle_radius=1),
                                                connection_drawing_spec= self.mp_draw.DrawingSpec(color=(0,0,255), thickness=1))
            except:
                pass
        return self.image_RGB
    

    def find_positons(self, image):
        list_points = []
        height, width, _ = image.shape
        try:
            for face_landmarks in self.results.multi_face_landmarks:
                face = face_landmarks
                for id_point, coordinates_xyz in enumerate(face.landmark):
                    list_points.append([id_point, int(coordinates_xyz.x*height), int(coordinates_xyz.y*width)])
        except:
            pass
        return list_points
    

    def draw_by_id(self, image, id):
        height, width, _ = image.shape
        try:
            for face_landmarks in self.results.multi_face_landmarks:
                face = face_landmarks
                for id_point, coordinates_xyz in enumerate(face.landmark):
                    if id_point == id:
                        coordinate_point = self.mp_draw._normalized_to_pixel_coordinates(coordinates_xyz.x, coordinates_xyz.y, width, height)
                        cv2.circle(image,coordinate_point,5,(255,0,0),-1)
        except:
            pass
        return image
    

def main():
    cap = cv2.VideoCapture(0)
    facemesh = face_mesh_detection()
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print('break frame null')
            continue
        image = facemesh.create_mesh(image)
        list_points = facemesh.find_positons(image)
        image = facemesh.draw_by_id(image, random.randint(0,len(list_points)))     
        cv2.imshow("image", image)
        #Press [Esc] for exit
        if cv2.waitKey(1)==27:
            break        
    if not cap.isOpened():
        print("Camera not found")
    cap.release()
    cv2.destroyAllWindows()       

    
if __name__=="__main__":
    main()
