import streamlit as st
import os
import numpy as np
import cv2
import time
from PIL import Image
import base64
from streamlit_option_menu import option_menu
from ultralytics import YOLO
import shutil
import tensorflow as tf
import cv2
import numpy as np
from test_yolo_poseEstimation import extract_frames
from test_yolo_poseEstimation import load_model
from moviepy.video.io.VideoFileClip import VideoFileClip
import subprocess
import imageio
from PIL import Image
import shutil
import os
import shutil

def delete_folder(folder_path):
    try:
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' successfully deleted.")
    except Exception as e:
        print(f"Error: {e}")

def convert_avi_to_mp4(input_path, output_folder):
    # Get the filename from the input_path
    file_name = os.path.splitext(os.path.basename(input_path))[0]
    
    # Construct the full output path with the same filename but with .mp4 extension
    output_path = os.path.join(output_folder, file_name + '.mp4')

    reader = imageio.get_reader(input_path)
    fps = reader.get_meta_data()['fps']
    writer = imageio.get_writer(output_path, fps=fps)

    for frame in reader:
        writer.append_data(frame)

    writer.close()

    # Return the full path of the generated MP4 file
    return output_path



@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file) 
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: scroll; # doesn't work
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return


#MenuBar
def streamlit_menu():
        # horizontal menu with custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["Home", "Video Analysis", "Live Stream"],  # required
            #icons=["house", "book", "envelope"],  # optional "icon": {"color": "orange", "font-size": "25px"},
            #menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "black"},
                
                "nav-link": {
                    "font-size": "18px",
                    "text-align": "left",
                    "color": "white",
                    "margin": "0px",
                    "--hover-color": "rgba(255, 0, 0, 0.7)",
                },
                "nav-link-selected": {"background-color": "black", "color": "rgba(255, 0, 0, 0.7)"},
            },
        )
        return selected


selected = streamlit_menu()

if selected == "Video Analysis":
        mp4_path = 'C:/Users/Lenovo/runs/pose/predict28/'
        delete_folder(mp4_path)
        set_png_as_page_bg('background1 (1).webp')
        coll1, coll2, coll3 = st.columns(3)
        with coll2:
            image = Image.open('logo.png')
            st.image(image)

        st.markdown(f'<p style=" font-family:Helvetica;  font-size:24px;  color: #9fcbcf;">| Drop your video below and let us uncover the hidden anomalies together</p>', unsafe_allow_html=True)

        with st.form('Upload', clear_on_submit=True):
            uploaded_file = st.file_uploader("Upload a Video", ['mp4','mov', 'avi'])
            uploaded = st.form_submit_button("Upload")
                
            
        if uploaded_file and uploaded:
            tfile = uploaded_file.read()
            tfile = uploaded_file.name
            tfile  = str(tfile)
            st.markdown(tfile)

            
            mdl = YOLO('yolov8n-pose.pt')  # load an official model
            results = mdl(source=tfile, show=False, conf=0.3, save=True)  
            
            file_name = os.path.splitext(tfile)[0]
            TestVideo = 'C:/Users/Lenovo/runs/pose/predict28/'+ file_name + '.avi'
            mp4_path = 'C:/Users/Lenovo/runs/pose/predict28/'
            result_mp4_path = convert_avi_to_mp4(TestVideo, mp4_path)
            model = load_model()
            input_frames = extract_frames(TestVideo)
            values = []
            seq = 0
            behavior ='None'
            st.markdown(f'<p style=" font-family:Helvetica;  font-size:24px; font-weight: bold; color: Blue;">Result :</p>', unsafe_allow_html=True)
            video_file = open(result_mp4_path, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
            for frames in input_frames:
                frames = np.expand_dims(frames, axis=0)
                predictions = model.predict(frames)
                predicted_class = np.argmax(predictions, axis=1)
                if np.max(predictions) < 0.5:
                    value_class = 7
                else:
                    value_class = predicted_class[0]
                
                if value_class == 0:
                    behavior = 'Hand shaking'
                elif value_class == 1:
                    behavior = 'Hugging'
                elif value_class == 2:
                    behavior = 'kicking'
                elif value_class == 3:
                    behavior = 'Pointing'  
                elif value_class == 4:
                    behavior = 'Punching'
                elif value_class == 5:
                    behavior = 'Pushing'
                else:
                    behavior = 'No Behavior Detected'
                print('the value is :',value_class)
                st.markdown(f'<p style=" font-family:Helvetica; font-weight: bold; color: white;">The identified behavior in sequence {seq} is: <span style=" font-family:Helvetica; font-weight: bold; color: yellow;">{behavior}</span></p>', unsafe_allow_html=True)
                values.append(value_class)
                seq += 1
            

if selected == "Home":
        set_png_as_page_bg('background1 (1).webp')
        coll1, coll2, coll3 = st.columns(3)
        with coll2:
            image = Image.open('logo.png')
            st.image(image)
           
        st.markdown('<h2 style="color:#9fcbcf; font-size: 28px;">| Spotting Unusual, Safeguarding the Usual: Abnormal Behavior Detection at Its Best!</h2>', unsafe_allow_html=True)
        st.markdown('<h3 style="color:gray; font-size: 20px">Experience the future of security, where innovation meets responsibility, only with VigilNet </h3>', unsafe_allow_html=True)
        st.markdown('<h3 style="color:white;  margin-bottom: 18px; border: 4px #000; border-radius: 50px; padding: 20px; background-color: rgba(3, 123, 252, 0.5); font-size: 18px">Welcome to VigilNet, where we redefine security with the power of cutting-edge AI technologies. Our intelligent solutions utilize state-of-the-art artificial intelligence to transform ordinary cameras into vigilant guardians of safety. By harnessing the prowess of AI, VigilNet goes beyond traditional surveillance, specializing in the detection of unusual behaviors such as Kicking,	Pointing,	Punching and Pushing.</h3>', unsafe_allow_html=True)
        #font-weight: bold; Our intelligent solutions utilize state-of-the-art artificial intelligence to transform ordinary cameras into vigilant guardians of safety. By harnessing the prowess of AI, VigilNet goes beyond traditional surveillance, specializing in the detection of unusual behaviors such as Kicking,	Pointing,	Punching and Pushing. Our systems are not just watchers; they are proactive guardians that send instant notifications to responsible authorities, ensuring swift response and resolution. With VigilNet, safety is not just observed; it is established, reported, and acted upon promptly. 
        # Create two columns
        col1, col2, col3, col4, col6,  col5 = st.columns(6)
        with col1:
            file_ = open("./img/dance1.gif", "rb")
            contents = file_.read()
            data_url = base64.b64encode(contents).decode("utf-8")
            file_.close()

            st.markdown(
                f'<img src="data:image/gif;base64,{data_url}" alt="gif" width=200 height=200>',
                unsafe_allow_html=True,
            )
        with col2:
            file_ = open("./img/openpose_amass_jump.gif", "rb")
            contents = file_.read()
            data_url = base64.b64encode(contents).decode("utf-8")
            file_.close()

            st.markdown(
                f'<img src="data:image/gif;base64,{data_url}" alt="gif" width=200 height=200>',
                unsafe_allow_html=True,
            )
        with col3:
            file_ = open("./img/openpose_amass_spin.gif", "rb")
            contents = file_.read()
            data_url = base64.b64encode(contents).decode("utf-8")
            file_.close()

            st.markdown(
                f'<img src="data:image/gif;base64,{data_url}" alt="gif" width=200 height=200>',
                unsafe_allow_html=True,
            )
        with col4:
            file_ = open("./img/openpose_walk_right.gif", "rb")
            contents = file_.read()
            data_url = base64.b64encode(contents).decode("utf-8")
            file_.close()

            st.markdown(
                f'<img src="data:image/gif;base64,{data_url}" alt="gif" width=200 height=200>',
                unsafe_allow_html=True,
            )
        with col6:
            file_ = open("./img/openpose_amass_stumble_away.gif", "rb")
            contents = file_.read()
            data_url = base64.b64encode(contents).decode("utf-8")
            file_.close()

            st.markdown(
                f'<img src="data:image/gif;base64,{data_url}" alt="gif" width=100 height=200>',
                unsafe_allow_html=True,
            )
        with col5:
            file_ = open("./img/openpose_arms_out.gif", "rb")
            contents = file_.read()
            data_url = base64.b64encode(contents).decode("utf-8")
            file_.close()

            st.markdown(
                f'<img src="data:image/gif;base64,{data_url}" alt="gif" width=100 height=200>',
                unsafe_allow_html=True,
            )


if selected == "Live Stream":
        
        set_png_as_page_bg('background1 (1).webp')
        coll1, coll2, coll3 = st.columns(3)
        with coll2:
            image = Image.open('logo.png')
            st.image(image)

        st.markdown(f'<p style=" font-family:Helvetica;  font-size:22px;  color: #9fcbcf;">| Start your Live Stream and let us uncover the Hidden Anomalies together</p>', unsafe_allow_html=True)
        
        run = st.checkbox('Run Live Stream')
        FRAME_WINDOW = st.image([])
        camera = cv2.VideoCapture(0)
        count = 0
        frames = []
        seq = 0
        start_time = time.time()
        model = load_model()
        while run:
            _, frame = camera.read()
            count += 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res_frame = cv2.resize(frame, (224, 224))

            # Load a model
            #model = YOLO('yolov8n-pose.pt')  # load an official model
            # Predict with the model
            #results = model(source=frame, show=False, conf=0.3, save=True)           
            frames.append(res_frame)
            FRAME_WINDOW.image(frame)
            if len(frames) == 10:
                    frames = np.expand_dims(frames, axis=0)    
                    predictions = model.predict(frames)
                    predicted_class = np.argmax(predictions, axis=1)
                    if np.max(predictions) < 0.5:
                        value_class = 7
                    else:
                        value_class = predicted_class[0]
                
                    if value_class == 0:
                        behavior = 'Hand shaking'
                    elif value_class == 1:
                        behavior = 'Hugging'
                    elif value_class == 2:
                        behavior = 'kicking'
                    elif value_class == 3:
                        behavior = 'Pointing'  
                    elif value_class == 4:
                        behavior = 'Punching'
                    elif value_class == 5:
                        behavior = 'Pushing'
                    else:
                        behavior = 'No Behavior Detected'
                    print('the value is :',value_class)
                    st.markdown(f'<p style=" font-family:Helvetica; font-weight: bold; color: white;">The identified behavior in sequence {seq} is: <span style=" font-family:Helvetica; font-weight: bold; color: yellow;">{behavior}</span></p>', unsafe_allow_html=True)
                    seq += 1
                    frames = []
            
        else:
            st.write('Stopped')    