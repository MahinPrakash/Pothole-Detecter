import streamlit as st
import time
import os
from ultralytics import YOLO
import cv2
import numpy as np

with open('styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
flag=0
uploaded_file = st.file_uploader("Upload a Video")
if uploaded_file is not None:
    cwd = os.getcwd()
    with open(os.path.join(cwd, 'uploaded_video.mp4'), 'wb') as f:
        f.write(uploaded_file.getbuffer())
        flag=1
with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("Pothole Detector")
    choice = st.radio("Navigation", ["Upload","Predict"])

if choice=='Predict':
  if(flag==1):
    video_path = 'uploaded_video.mp4'
    video_path_out = 'output.mp4'

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    H, W, _ = frame.shape
    out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'avc1'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

    model = YOLO('last.pt') 

    threshold = 0.5
    
    st.write("Making Detections...")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]

        for result in results.boxes.data.tolist():

            x1, y1, x2, y2, score, class_id = result
            if score > threshold:

                class_name = results.names[int(class_id)].upper()

                box_color = (0, 255, 0)  
                if class_name == 'SMALL':
                    box_color = (0, 255, 0) 
                elif class_name == 'MEDIUM':
                    box_color = (255, 255, 0)  
                elif class_name == 'LARGE':
                    box_color = (255, 0, 0)  

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 4)

                cv2.putText(frame, class_name, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, box_color, 3, cv2.LINE_AA)

        out.write(frame)
        progress_bar.progress((i+1)/total_frames)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    st.success('Detections are done!', icon="✅")
    st.subheader('Result:-')
    video_file = open(video_path_out, 'rb') 
    video_bytes = video_file.read() 
    st.video(video_bytes) 



