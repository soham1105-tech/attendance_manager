import os
import dlib
import csv
import numpy as np
import cv2
import streamlit as st
import pandas as pd
import time

# --- 1. CONFIGURATION AND MODEL LOADING ---

path_images_from_camera = "data/data_faces_from_camera/"
csv_file_path = "data/features_all.csv"

# Use st.cache_resource to load these heavy models only once
@st.cache_resource
def load_dlib_models():
    """Loads Dlib models and returns them."""
    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
        face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")
        return detector, predictor, face_reco_model
    except RuntimeError as e:
        st.error(f"Error loading Dlib models: {e}")
        st.error("Please make sure 'shape_predictor_68_face_landmarks.dat' and "
                 "'dlib_face_recognition_resnet_model_v1.dat' are in the 'data/data_dlib/' directory.")
        return None, None, None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading models: {e}")
        return None, None, None

# Load models
detector, predictor, face_reco_model = load_dlib_models()

# --- 2. CORE FEATURE EXTRACTION LOGIC (from your script) ---

def return_128d_features(path_img, log_container):
    """
    Return 128D features for single image.
    Logs output to the Streamlit log_container.
    """
    if not (detector and predictor and face_reco_model):
        return 0  # Models failed to load

    img_rd = cv2.imread(path_img)
    if img_rd is None:
        log_container.warning(f"Could not read image: {path_img}")
        return 0
        
    faces = detector(img_rd, 1)
    
    if len(faces) != 0:
        shape = predictor(img_rd, faces[0])
        face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
    else:
        face_descriptor = 0
        log_container.warning(f"No face detected in: {os.path.basename(path_img)}")
    return face_descriptor

def return_features_mean_personX(path_face_personX, log_container, img_progress_bar):
    """
    Return the mean value of 128D face descriptor for person X.
    Updates progress bars in the Streamlit UI.
    """
    features_list_personX = []
    photos_list = os.listdir(path_face_personX)
    
    if not photos_list:
        log_container.warning(f"Warning: No images in {os.path.basename(path_face_personX)}/")
        return np.zeros(128, dtype=object, order='C')

    for i, photo in enumerate(photos_list):
        # Update image processing progress
        img_progress_bar.progress((i + 1) / len(photos_list), text=f"Processing image {i+1}/{len(photos_list)}")
        
        # Log which image is being read
        log_container.info(f"Reading image: {photo}")
        
        features_128d = return_128d_features(os.path.join(path_face_personX, photo), log_container)
        
        # Jump if no face detected from image
        if features_128d != 0:
            features_list_personX.append(features_128d)

    if features_list_personX:
        features_mean_personX = np.array(features_list_personX, dtype=object).mean(axis=0)
    else:
        log_container.error(f"No faces were detected in any images for {os.path.basename(path_face_personX)}.")
        features_mean_personX = np.zeros(128, dtype=object, order='C')
        
    return features_mean_personX

def run_feature_extraction(person_list, log_container, overall_progress_bar):
    """
    Main processing loop. Iterates through all person folders and saves features to CSV.
    Returns the data for display in Streamlit.
    """
    all_features_data = []
    
    with open(csv_file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        
        for i, person_folder in enumerate(person_list):
            log_container.markdown(f"--- \n ### Processing: **{person_folder}**")
            
            # Placeholder for the per-person image progress bar
            img_progress_placeholder = log_container.empty()
            img_progress_bar = img_progress_placeholder.progress(0)
            
            # Get the mean features for this person
            features_mean_personX = return_features_mean_personX(
                os.path.join(path_images_from_camera, person_folder), 
                log_container, 
                img_progress_bar
            )
            
            # Clear the per-person progress bar
            img_progress_placeholder.empty()

            # Get person name from folder name (e.g., "person_1_tom" -> "tom")
            if len(person_folder.split('_', 2)) == 2:
                person_name = person_folder  # "person_x"
            else:
                person_name = person_folder.split('_', 2)[-1]  # "person_x_name"
            
            # Insert person name as the first column
            features_with_name = np.insert(features_mean_personX, 0, person_name, axis=0)
            
            # Write to CSV
            writer.writerow(features_with_name)
            
            # Add to list for final display
            all_features_data.append(features_with_name)
            
            # Update overall progress
            overall_progress_bar.progress((i + 1) / len(person_list), text=f"Overall Progress: {i+1}/{len(person_list)} persons")
            
            log_container.success(f"Finished processing: {person_folder}")

    return all_features_data

# --- 3. STREAMLIT UI ---

st.set_page_config(page_title="Face Feature Extractor", layout="wide")
st.title("👨‍💻 Face Feature Extractor")
st.write(f"""
This app processes all face images stored in `{path_images_from_camera}`. 
It calculates a 128-dimension feature vector for each person and saves the result in `{csv_file_path}`.
""")

# Stop if models failed to load
if not (detector and predictor and face_reco_model):
    st.error("Dlib models failed to load. The app cannot continue.")
    st.stop()

# Check for data directory
if not os.path.isdir(path_images_from_camera):
    st.error(f"Error: Directory not found: `{path_images_from_camera}`")
    st.warning("Please run the 'Face Register' app first to save some face images.")
    st.stop()

# Find person folders
try:
    person_list = [f for f in os.listdir(path_images_from_camera) 
                   if os.path.isdir(os.path.join(path_images_from_camera, f))]
    person_list.sort()
    
    if not person_list:
        st.warning(f"No person folders found in `{path_images_from_camera}`. Please register faces first.")
        st.stop()
        
    st.subheader(f"Found {len(person_list)} person(s) to process:")
    st.text('\n'.join(person_list))

except FileNotFoundError:
    st.error(f"Directory not found: {path_images_from_camera}")
    st.stop()


st.markdown("---")

# Button to start the process
if st.button("🚀 Start Feature Extraction", type="primary", use_container_width=True):
    
    st.subheader("📊 Overall Progress")
    overall_progress_bar = st.progress(0, text="Starting...")
    
    st.subheader("Processing Log")
    log_expander = st.expander("Show processing details", expanded=True)
    log_container = log_expander.container(height=300)
    
    start_time = time.time()
    
    with st.spinner("Processing faces... This might take a few minutes."):
        all_features_data = run_feature_extraction(person_list, log_container, overall_progress_bar)
    
    end_time = time.time()
    
    st.success(f"✅ **Processing Complete!**")
    st.info(f"Total time taken: {end_time - start_time:.2f} seconds")
    st.balloons()
    
    # Display the results in a DataFrame
    st.subheader("Generated Features")
    try:
        header = ["person_name"] + [f"feature_{i}" for i in range(128)]
        df = pd.DataFrame(all_features_data, columns=header)
        st.dataframe(df)
    except Exception as e:
        st.error(f"Could not display features in DataFrame: {e}")

st.markdown("---")

# Section to show the *current* CSV file, if it exists
st.subheader("Current `features_all.csv` File")
if os.path.isfile(csv_file_path):
    try:
        df_existing = pd.read_csv(csv_file_path, header=None)
        if df_existing.empty:
            st.warning("`features_all.csv` is currently empty.")
        else:
            st.info(f"Displaying existing data from `{csv_file_path}`:")
            # Add header for display
            header = ["person_name"] + [f"feature_{i}" for i in range(128)]
            if len(df_existing.columns) == 129:
                df_existing.columns = header
            st.dataframe(df_existing)
    except pd.errors.EmptyDataError:
        st.warning("`features_all.csv` is currently empty.")
    except Exception as e:
        st.error(f"Could not read or display `{csv_file_path}`: {e}")
else:
    st.info(f"`{csv_file_path}` does not exist yet. Run the extraction to create it.")