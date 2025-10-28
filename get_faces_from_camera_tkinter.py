import dlib
import numpy as np
import cv2
import os
import shutil
import time
import logging
import tkinter as tk
from tkinter import ttk 
from tkinter import font as tkFont
from PIL import Image, ImageTk

# Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()


class Face_Register:
    def __init__(self):

        self.current_frame_faces_cnt = 0
        self.existing_faces_cnt = 0
        self.ss_cnt = 0

        # Tkinter GUI
        self.win = tk.Tk()
        self.win.title("Face Register")
        
        # --- GEOMETRY AND AESTHETIC SETUP ---
        
        # Define Aesthetic Grey Colors and Font
        self.main_grey = '#EBEBEB'     # Light/Platinum Grey - Main UI background
        self.shadow_color = '#BBBCB6'  # Slightly darker Grey - Right panel background (The "shadow" effect)
        self.sans_serif_font = 'Helvetica'
        self.circular_button_bg = '#778899' # Slate Grey for the "Clear" button

        # Set window size to ensure ample space for 640px camera and 400px panel
        self.WINDOW_WIDTH = 1060 # 640 + 420
        self.win.geometry(f"{self.WINDOW_WIDTH}x500") 
        self.win.configure(bg=self.main_grey) 

        # 2. Setup ttk Style
        self.style = ttk.Style()
        self.style.theme_use('clam') 

        # Configure styles for modern, clean text/labels (using sans-serif font)
        self.style.configure('Aesthetic.TFrame', 
                             background=self.shadow_color, 
                             relief='flat', 
                             borderwidth=0)

        self.style.configure('Aesthetic.TLabel', 
                             background=self.shadow_color, 
                             foreground='#333333', 
                             font=(self.sans_serif_font, 12)) 
        
        self.style.configure('AestheticHeader.TLabel', 
                             background=self.shadow_color, 
                             foreground='#1A1A1A', 
                             font=(self.sans_serif_font, 20, 'bold'))

        # Configure Aesthetic Button style 
        self.style.configure('Aesthetic.TButton', 
                             font=(self.sans_serif_font, 12), 
                             padding=[10, 5], 
                             relief="flat", 
                             background='#CCCCCC',
                             foreground='#333333')
        self.style.map('Aesthetic.TButton', background=[('active', '#D9D9D9')])


        # Configure Circular button style
        self.style.configure('Circular.TButton', 
                             font=(self.sans_serif_font, 12, 'bold'), 
                             padding=10, 
                             relief="flat", 
                             background=self.circular_button_bg, 
                             foreground='white',
                             bordercolor=self.circular_button_bg) 
        self.style.map('Circular.TButton', background=[('active', '#5A6371')])
        
        # --- END OF STYLING ---

        # GUI left part (Camera)
        self.frame_left_camera = tk.Frame(self.win)
        self.label = tk.Label(self.win)
        self.label.pack(side=tk.LEFT)
        self.frame_left_camera.pack()

        # GUI right part (Aesthetic Panel)
        self.frame_right_info = ttk.Frame(self.win, style='Aesthetic.TFrame', padding=(10, 10))
        self.frame_right_info.pack(side=tk.LEFT, fill=tk.BOTH, expand=True) 

        
        # Initialize all necessary Labels and Entry fields 
        bg_color = self.shadow_color
        
        self.label_cnt_face_in_database = tk.Label(self.frame_right_info, text=str(self.existing_faces_cnt), bg=bg_color, font=(self.sans_serif_font, 12))
        self.label_fps_info = tk.Label(self.frame_right_info, text="", bg=bg_color, font=(self.sans_serif_font, 12))
        self.input_name = tk.Entry(self.frame_right_info, font=(self.sans_serif_font, 12))
        self.input_name_char = ""
        self.label_warning = tk.Label(self.frame_right_info, bg=bg_color, fg='red', font=(self.sans_serif_font, 12, 'bold'))
        self.label_face_cnt = tk.Label(self.frame_right_info, text="", bg=bg_color, font=(self.sans_serif_font, 12))
        self.log_all = tk.Label(self.frame_right_info, bg=bg_color, font=(self.sans_serif_font, 10), justify=tk.LEFT, wraplength=400)

        self.font_title = tkFont.Font(family=self.sans_serif_font, size=20, weight='bold')
        self.font_step_title = tkFont.Font(family=self.sans_serif_font, size=15, weight='bold')
        self.font_warning = tkFont.Font(family=self.sans_serif_font, size=15, weight='bold')

        self.path_photos_from_camera = "data/data_faces_from_camera/"
        self.current_face_dir = ""
        self.font = cv2.FONT_ITALIC

        # Current frame and face ROI position (kept for class methods)
        self.current_frame = np.ndarray
        self.face_ROI_image = np.ndarray
        self.face_ROI_width_start = 0
        self.face_ROI_height_start = 0
        self.face_ROI_width = 0
        self.face_ROI_height = 0
        self.ww = 0
        self.hh = 0

        self.out_of_range_flag = False
        self.face_folder_created_flag = False

        # FPS
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

        self.cap = cv2.VideoCapture(0)

    # Delete old face folders
    def GUI_clear_data(self):
        folders_rd = os.listdir(self.path_photos_from_camera)
        for i in range(len(folders_rd)):
            shutil.rmtree(self.path_photos_from_camera + folders_rd[i])
        if os.path.isfile("data/features_all.csv"):
            os.remove("data/features_all.csv")
        self.label_cnt_face_in_database['text'] = "0"
        self.existing_faces_cnt = 0
        self.log_all["text"] = "Face images and `features_all.csv` removed!"

    def GUI_get_input_name(self):
        self.input_name_char = self.input_name.get()
        self.create_face_folder()
        self.label_cnt_face_in_database['text'] = str(self.existing_faces_cnt)

    def GUI_info(self):
        # --- FINAL LAYOUT FIX: Robust Grid Structure ---
        PAD_X_MAIN = 15
        PAD_X_SMALL = 5
        
        # Configure column weights: Col 1 (Entry/Values) expands to fill available space
        self.frame_right_info.grid_columnconfigure(0, weight=0) # Labels (Fixed)
        self.frame_right_info.grid_columnconfigure(1, weight=1) # Values/Entry (Expands)
        self.frame_right_info.grid_columnconfigure(2, weight=0) # Button (Fixed)

        # Row 0: Title (Spans all 3 columns)
        ttk.Label(self.frame_right_info,
                  text="Face register",
                  style='AestheticHeader.TLabel').grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=PAD_X_MAIN, pady=25)
        
        # Info Block: FPS, Database, Current Frame
        info_labels = [
            ("FPS:", self.label_fps_info),
            ("Faces in database:", self.label_cnt_face_in_database),
            ("Faces in current frame:", self.label_face_cnt)
        ]
        
        # Rows 1, 2, 3: Info Lines
        for i, (text, value_widget) in enumerate(info_labels):
            ttk.Label(self.frame_right_info, text=text, style='Aesthetic.TLabel').grid(
                row=i + 1, column=0, sticky=tk.W, padx=PAD_X_MAIN, pady=2)
            value_widget.grid(
                row=i + 1, column=1, columnspan=2, sticky=tk.W, padx=0, pady=2) 

        # Row 4: Warning (Spans all columns)
        self.label_warning.grid(row=4, column=0, columnspan=3, sticky=tk.W, padx=PAD_X_MAIN, pady=5) 

        # Row 5: Step 1 Title
        ttk.Label(self.frame_right_info,
                  font=self.font_step_title,
                  text="Step 1: Clear face photos", style='Aesthetic.TLabel').grid(row=5, column=0, columnspan=3, sticky=tk.W, padx=PAD_X_MAIN, pady=20)
        
        # Row 6: Clear button (Spans all columns)
        ttk.Button(self.frame_right_info,
                   text='Clear',
                   command=self.GUI_clear_data,
                   style='Circular.TButton',
                   width=8).grid(row=6, column=0, columnspan=3, sticky=tk.W, padx=PAD_X_MAIN, pady=2)

        # Row 7: Step 2 Title
        ttk.Label(self.frame_right_info,
                  font=self.font_step_title,
                  text="Step 2: Input name", style='Aesthetic.TLabel').grid(row=7, column=0, columnspan=3, sticky=tk.W, padx=PAD_X_MAIN, pady=20)

        # Row 8: Name Input Row
        ttk.Label(self.frame_right_info, text="Name: ", style='Aesthetic.TLabel').grid(row=8, column=0, sticky=tk.W, padx=PAD_X_MAIN, pady=0)
        
        # Input Entry field (stretches using sticky W+E in the expandable Col 1)
        self.input_name.grid(row=8, column=1, sticky=tk.W+tk.E, padx=0, pady=2) 

        # Input Button in Col 2
        ttk.Button(self.frame_right_info,
                   text='Input',
                   style='Aesthetic.TButton',
                   width=5,
                   command=self.GUI_get_input_name).grid(row=8, column=2, sticky=tk.W, padx=PAD_X_SMALL)

        # Row 9: Step 3 Title
        ttk.Label(self.frame_right_info,
                  font=self.font_step_title,
                  text="Step 3: Save face image", style='Aesthetic.TLabel').grid(row=9, column=0, columnspan=3, sticky=tk.W, padx=PAD_X_MAIN, pady=20)

        # Row 10: Save Button (Spans all columns)
        ttk.Button(self.frame_right_info,
                   text='Save current face',
                   style='Aesthetic.TButton',
                   command=self.save_current_face).grid(row=10, column=0, columnspan=3, sticky=tk.W, padx=PAD_X_MAIN)

        # Row 11: Log Message (Spans all columns)
        self.log_all.grid(row=11, column=0, columnspan=3, sticky=tk.W, padx=PAD_X_MAIN, pady=20) 

    # Mkdir for saving photos and csv
    def pre_work_mkdir(self):
        # Create folders to save face images and csv
        if os.path.isdir(self.path_photos_from_camera):
            pass
        else:
            os.mkdir(self.path_photos_from_camera)

    # Start from person_x+1
    def check_existing_faces_cnt(self):
        if os.listdir("data/data_faces_from_camera/"):
            # Get the order of latest person
            person_list = os.listdir("data/data_faces_from_camera/")
            person_num_list = []
            for person in person_list:
                person_order = person.split('_')[1].split('_')[0]
                person_num_list.append(int(person_order))
            self.existing_faces_cnt = max(person_num_list)

        # Start from person_1
        else:
            self.existing_faces_cnt = 0

    # Update FPS of Video stream
    def update_fps(self):
        now = time.time()
        # Refresh fps per second
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

        self.label_fps_info["text"] = str(self.fps.__round__(2))

    def create_face_folder(self):
        # Create the folders for saving faces
        self.existing_faces_cnt += 1
        if self.input_name_char:
            self.current_face_dir = self.path_photos_from_camera + \
                                     "person_" + str(self.existing_faces_cnt) + "_" + \
                                     self.input_name_char
        else:
            self.current_face_dir = self.path_photos_from_camera + \
                                     "person_" + str(self.existing_faces_cnt)
        os.makedirs(self.current_face_dir)
        self.log_all["text"] = "\"" + self.current_face_dir + "/\" created!"
        logging.info("\n%-40s %s", "Create folders:", self.current_face_dir)

        self.ss_cnt = 0  # Clear the cnt of screen shots
        self.face_folder_created_flag = True  # Face folder already created

    def save_current_face(self):
        if self.face_folder_created_flag:
            if self.current_frame_faces_cnt == 1:
                if not self.out_of_range_flag:
                    self.ss_cnt += 1
                    # Create blank image according to the size of face detected
                    self.face_ROI_image = np.zeros((int(self.face_ROI_height * 2), self.face_ROI_width * 2, 3),
                                                    np.uint8)
                    for ii in range(self.face_ROI_height * 2):
                        for jj in range(self.face_ROI_width * 2):
                            self.face_ROI_image[ii][jj] = self.current_frame[self.face_ROI_height_start - self.hh + ii][
                                 self.face_ROI_width_start - self.ww + jj]
                    self.log_all["text"] = "\"" + self.current_face_dir + "/img_face_" + str(
                        self.ss_cnt) + ".jpg\"" + " saved!"
                    self.face_ROI_image = cv2.cvtColor(self.face_ROI_image, cv2.COLOR_BGR2RGB)

                    cv2.imwrite(self.current_face_dir + "/img_face_" + str(self.ss_cnt) + ".jpg", self.face_ROI_image)
                    logging.info("%-40s %s/img_face_%s.jpg", "Save into：",
                                  str(self.current_face_dir), str(self.ss_cnt) + ".jpg")
                else:
                    self.log_all["text"] = "Please do not out of range!"
            else:
                self.log_all["text"] = "No face in current frame!"
        else:
            self.log_all["text"] = "Please run step 2!"

    def get_frame(self):
        try:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                frame = cv2.resize(frame, (640,480))
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except:
            print("Error: No video input!!!")
            return False, None

    # Main process of face detection and saving
    def process(self):
        ret, self.current_frame = self.get_frame()
        
        if ret is False and self.current_frame is None:
            self.win.after(2000, self.process)
            return

        faces = detector(self.current_frame, 0)
        
        if ret:
            self.update_fps()
            self.label_face_cnt["text"] = str(len(faces))
            
            if len(faces) != 0:
                for k, d in enumerate(faces):
                    self.face_ROI_width_start = d.left()
                    self.face_ROI_height_start = d.top()
                    self.face_ROI_height = (d.bottom() - d.top())
                    self.face_ROI_width = (d.right() - d.left())
                    self.hh = int(self.face_ROI_height / 2)
                    self.ww = int(self.face_ROI_width / 2)

                    # Determine color for the bounding box
                    if (d.right() + self.ww) > 640 or (d.bottom() + self.hh > 480) or (d.left() - self.ww < 0) or (
                            d.top() - self.hh < 0):
                        self.label_warning["text"] = "OUT OF RANGE"
                        self.label_warning['fg'] = 'red'
                        self.out_of_range_flag = True
                        color_rectangle = (0, 0, 255) # BGR for Red (for warning)
                    else:
                        self.out_of_range_flag = False
                        self.label_warning["text"] = ""
                        # Deep Blue for the bounding box in normal state
                        color_rectangle = (255, 0, 0) # BGR for Deep Blue 
                    
                    self.current_frame = cv2.rectangle(self.current_frame,
                                                         tuple([d.left() - self.ww, d.top() - self.hh]),
                                                         tuple([d.right() + self.ww, d.bottom() + self.hh]),
                                                         color_rectangle, 2)
            self.current_frame_faces_cnt = len(faces)

            img_Image = Image.fromarray(self.current_frame)
            img_PhotoImage = ImageTk.PhotoImage(image=img_Image)
            self.label.img_tk = img_PhotoImage
            self.label.configure(image=img_PhotoImage)

        self.win.after(20, self.process)

    def run(self):
        self.pre_work_mkdir()
        self.check_existing_faces_cnt()
        self.GUI_info()
        self.process()
        self.win.mainloop()


def main():
    if not os.path.isdir("data"):
        os.mkdir("data")
    if not os.path.isdir("data/data_faces_from_camera"):
        os.mkdir("data/data_faces_from_camera")
        
    logging.basicConfig(level=logging.INFO)
    Face_Register_con = Face_Register()
    Face_Register_con.run()


if __name__ == '__main__':
    main()