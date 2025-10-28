import dlib
import numpy as np
import cv2
import os
import pandas as pd
import time
import logging
import sqlite3
import datetime


# Dlib  / Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

# Dlib landmark / Get face landmarks
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

# Dlib Resnet Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

# Create a connection to the database
conn = sqlite3.connect("attendance.db")
cursor = conn.cursor()

# Create a table for the current date (Using fixed table name 'attendance')
table_name = "attendance" 
create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} (name TEXT, time TEXT, date DATE, UNIQUE(name, date))"
cursor.execute(create_table_sql)


# Commit changes and close the connection
conn.commit()
conn.close()


class Face_Recognizer:
    def __init__(self):
        # Reverting to original font
        self.font = cv2.FONT_ITALIC 

        # FPS
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

        # cnt for frame
        self.frame_cnt = 0

        # Save the features of faces in the database
        self.face_features_known_list = []
        # / Save the name of faces in the database
        self.face_name_known_list = []

        # List to save centroid positions of ROI in frame N-1 and N
        self.last_frame_face_centroid_list = []
        self.current_frame_face_centroid_list = []

        # List to save names of objects in frame N-1 and N
        self.last_frame_face_name_list = [] 
        self.current_frame_face_name_list = []

        # cnt for faces in frame N-1 and N
        self.last_frame_face_cnt = 0
        self.current_frame_face_cnt = 0

        # Save the e-distance for faceX when recognizing
        self.current_frame_face_X_e_distance_list = []

        # Save the positions and names of current faces captured
        self.current_frame_face_position_list = []
        # Save the features of people in current frame
        self.current_frame_face_feature_list = []

        # e distance between centroid of ROI in last and current frame
        self.last_current_frame_centroid_e_distance = 0

        # Reclassify after 'reclassify_interval' frames
        self.reclassify_interval_cnt = 0
        self.reclassify_interval = 10


    # "features_all.csv"  / Get known faces from "features_all.csv"
    def get_face_database(self):
        path_features_known_csv = "data/features_all.csv"

        if os.path.exists(path_features_known_csv):
            # --- FIX: Handle EmptyDataError ---
            try:
                if os.path.getsize(path_features_known_csv) == 0:
                    logging.warning("'features_all.csv' is empty! Please register faces.")
                    return 0

                csv_rd = pd.read_csv(path_features_known_csv, header=None)
                
                if csv_rd.empty or csv_rd.shape[1] < 129:
                    logging.warning("Feature CSV found, but contains no recognizable data rows.")
                    return 0

                for i in range(csv_rd.shape[0]):
                    features_someone_arr = []
                    self.face_name_known_list.append(csv_rd.iloc[i][0])
                    for j in range(1, 129):
                        if pd.isna(csv_rd.iloc[i][j]) or csv_rd.iloc[i][j] == '': 
                            features_someone_arr.append('0')
                        else:
                            features_someone_arr.append(csv_rd.iloc[i][j])
                    self.face_features_known_list.append(features_someone_arr)
                
                logging.info("Faces in Database: %d", len(self.face_features_known_list))
                return 1
            
            except pd.errors.EmptyDataError:
                logging.warning("EmptyDataError: 'features_all.csv' has no columns/data. Please register faces!")
                return 0
            
            except Exception as e:
                logging.error(f"An unexpected error occurred while reading features_all.csv: {e}")
                return 0
            
        else:
            logging.warning("'features_all.csv' not found!")
            logging.warning("Please ensure faces have been registered.")
            return 0

    def update_fps(self):
        now = time.time()
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    @staticmethod
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1, dtype=float)
        feature_2 = np.array(feature_2, dtype=float)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    def centroid_tracker(self):
        for i in range(len(self.current_frame_face_centroid_list)):
            e_distance_current_frame_person_x_list = []
            for j in range(len(self.last_frame_face_centroid_list)):
                self.last_current_frame_centroid_e_distance = self.return_euclidean_distance(
                    self.current_frame_face_centroid_list[i], self.last_frame_face_centroid_list[j])

                e_distance_current_frame_person_x_list.append(
                    self.last_current_frame_centroid_e_distance)

            last_frame_num = e_distance_current_frame_person_x_list.index(
                min(e_distance_current_frame_person_x_list))
            self.current_frame_face_name_list[i] = self.last_frame_face_name_list[last_frame_num]

    # cv2 window / putText on cv2 window
    def draw_note(self, img_rd):
        # --- Reverting to original drawing logic ---
        # / Add some info on windows
        cv2.putText(img_rd, "Face Recognizer with Deep Learning", (20, 40), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Frame: " + str(self.frame_cnt), (20, 100), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "FPS: " + str(self.fps.__round__(2)), (20, 130), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Faces: " + str(self.current_frame_face_cnt), (20, 160), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Q: Quit", (20, img_rd.shape[0] - 20), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA) # Adjusted Q:Quit position
        # ------------------------------------------

        for i in range(len(self.current_frame_face_name_list)):
            img_rd = cv2.putText(img_rd, self.current_frame_face_name_list[i], tuple(
                [int(self.current_frame_face_centroid_list[i][0]) - 50, int(self.current_frame_face_centroid_list[i][1]) + 150]), # Used the original custom positioning
                                     self.font,
                                     0.8, (255, 190, 0),
                                     1,
                                     cv2.LINE_AA)
                                     
    # insert data in database
    def attendance(self, name):
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        
        # Check if the name already has an entry for the current date
        cursor.execute("SELECT * FROM attendance WHERE name = ? AND date = ?", (name, current_date))
        existing_entry = cursor.fetchone()

        if existing_entry:
            # --- FIX APPLIED HERE: Change to logging.debug ---
            # This message will still be logged but won't spam the terminal set to INFO level.
            logging.debug(f"{name} is already marked as present for {current_date}") 
            
        else:
            current_time = datetime.datetime.now().strftime('%H:%M:%S')
            # Use ON CONFLICT IGNORE for safety against duplicate insertion attempts
            cursor.execute("INSERT OR IGNORE INTO attendance (name, time, date) VALUES (?, ?, ?)", (name, current_time, current_date))
            conn.commit()
            # This is a new, important event, so keep it as logging.info
            logging.info(f"{name} marked as present for {current_date} at {current_time}")

        conn.close()

    def process(self, stream):
        # 1. Get faces known from "features.all.csv"
        if self.get_face_database():
            
            window_name = "camera" # Use the original window name
            
            # --- FULL SCREEN SETUP (Simplified) ---
            cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            # -------------------------

            while stream.isOpened():
                self.frame_cnt += 1
                flag, img_rd = stream.read()
                
                if not flag:
                    logging.warning("Could not read frame from stream. Exiting loop.")
                    break
                    
                # We need to resize the image before processing/drawing, otherwise the face detection
                # coordinates will be wrong relative to the window size.
                img_rd = cv2.resize(img_rd, (1280, 720)) # Using a standard HD resolution for better fullscreen viewing
                
                kk = cv2.waitKey(1)

                # 2. Detect faces for frame X
                faces = detector(img_rd, 0)

                # 3. Update cnt for faces in frames
                self.last_frame_face_cnt = self.current_frame_face_cnt
                self.current_frame_face_cnt = len(faces)

                # 4. Update the face name list in last frame
                self.last_frame_face_name_list = self.current_frame_face_name_list[:]

                # 5. update frame centroid list
                self.last_frame_face_centroid_list = self.current_frame_face_centroid_list
                self.current_frame_face_centroid_list = []

                # 6.1 if cnt not changes
                if (self.current_frame_face_cnt == self.last_frame_face_cnt) and (
                        self.reclassify_interval_cnt != self.reclassify_interval):
                    
                    self.current_frame_face_position_list = []

                    if "unknown" in self.current_frame_face_name_list:
                        self.reclassify_interval_cnt += 1

                    if self.current_frame_face_cnt != 0:
                        for k, d in enumerate(faces):
                            self.current_frame_face_position_list.append(tuple(
                                [d.left(), int(d.bottom() + (d.bottom() - d.top()) / 4)]))
                            self.current_frame_face_centroid_list.append(
                                [int(d.left() + d.right()) / 2,
                                 int(d.top() + d.bottom()) / 2])

                            # Reverting to the original white box
                            cv2.rectangle(img_rd,
                                             tuple([d.left(), d.top()]),
                                             tuple([d.right(), d.bottom()]),
                                             (255, 255, 255), 2)

                        # Multi-faces in current frame, use centroid-tracker to track
                        if self.current_frame_face_cnt != 1:
                            self.centroid_tracker()

                        for i in range(self.current_frame_face_cnt):
                            pass # Attendance handled in recognition phase
                            
                        self.draw_note(img_rd)
                        
                # 6.2 If cnt of faces changes, do full recognition
                else:
                    self.current_frame_face_position_list = []
                    self.current_frame_face_X_e_distance_list = []
                    self.current_frame_face_feature_list = []
                    self.reclassify_interval_cnt = 0

                    if self.current_frame_face_cnt == 0:
                        self.current_frame_face_name_list = []
                        
                    else:
                        self.current_frame_face_name_list = []
                        for i in range(len(faces)):
                            shape = predictor(img_rd, faces[i])
                            self.current_frame_face_feature_list.append(
                                face_reco_model.compute_face_descriptor(img_rd, shape))
                            self.current_frame_face_name_list.append("unknown")

                        for k, d in enumerate(faces):
                            self.current_frame_face_centroid_list.append(
                                [int(d.left() + d.right()) / 2,
                                 int(d.top() + d.bottom()) / 2])

                            self.current_frame_face_X_e_distance_list = []

                            self.current_frame_face_position_list.append(tuple(
                                [d.left(), int(d.bottom() + (d.bottom() - d.top()) / 4)]))
                            
                            # Draw white bounding box
                            cv2.rectangle(img_rd,
                                             tuple([d.left(), d.top()]),
                                             tuple([d.right(), d.bottom()]),
                                             (255, 255, 255), 2)

                            # 6.2.2.3 Compare with known faces
                            for i in range(len(self.face_features_known_list)):
                                known_features = np.array(self.face_features_known_list[i], dtype=float)
                                
                                if known_features[0] != 0.0:
                                    e_distance_tmp = self.return_euclidean_distance(
                                        self.current_frame_face_feature_list[k], known_features)
                                    self.current_frame_face_X_e_distance_list.append(e_distance_tmp)
                                else:
                                    self.current_frame_face_X_e_distance_list.append(999999999)

                            if self.current_frame_face_X_e_distance_list:
                                similar_person_num = self.current_frame_face_X_e_distance_list.index(
                                    min(self.current_frame_face_X_e_distance_list))

                                if min(self.current_frame_face_X_e_distance_list) < 0.4:
                                    name = self.face_name_known_list[similar_person_num]
                                    self.current_frame_face_name_list[k] = name
                                    
                                    # Insert attendance record
                                    self.attendance(name)
                                else:
                                    pass # Remains "unknown"
                        
                        self.draw_note(img_rd)


                # 8. 'q' / Press 'q' to exit
                if kk == ord('q'):
                    break

                self.update_fps()
                cv2.imshow(window_name, img_rd)


    def run(self):
        # Setting camera resolution to HD (1280x720) for better fullscreen quality
        cap = cv2.VideoCapture(0)
        
        # Set Resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.process(cap)

        cap.release()
        cv2.destroyAllWindows()
    
    


def main():
    if not os.path.isdir("data"):
        os.mkdir("data")
        
    logging.basicConfig(level=logging.INFO)
    Face_Recognizer_con = Face_Recognizer()
    Face_Recognizer_con.run()


if __name__ == '__main__':
    main()