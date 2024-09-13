import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
import tempfile
import pytesseract

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Set page config
st.set_page_config(page_title="License Plate Detection and Recognition", layout="wide")

# Set the title of the Streamlit app
st.title("License Plate Detection and Recognition")

# Load YOLO model
@st.cache_resource
def load_model():
    try:
        return YOLO('kbest.pt')
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

model = load_model()

def predict_and_recognize(image):
    try:
        results = model.predict(image, device='cpu')
        detected_plates = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                
                # Extract the license plate region
                plate_region = image[y1:y2, x1:x2]
                
                # Perform OCR on the plate region
                try:
                    plate_text = pytesseract.image_to_string(plate_region, config='--psm 7 --oem 3')
                    plate_text = plate_text.strip()
                except Exception as ocr_error:
                    st.warning(f"OCR Error: {ocr_error}")
                    plate_text = "OCR Failed"
                
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f'{plate_text} ({confidence*100:.2f}%)', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                
                detected_plates.append((plate_text, confidence))
        
        return image, detected_plates
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, []

def process_video(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error(f"Error opening video file: {video_path}")
            return None, []

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        temp_output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (frame_width, frame_height))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = st.progress(0)

        all_detected_plates = []

        for frame_num in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame, detected_plates = predict_and_recognize(frame)
            out.write(processed_frame)
            all_detected_plates.extend(detected_plates)
            progress_bar.progress((frame_num + 1) / total_frames)

        cap.release()
        out.release()
        return temp_output_path, all_detected_plates
    except Exception as e:
        st.error(f"Error processing video: {e}")
        return None, []

def cleanup_temp_files():
    temp_dir = tempfile.gettempdir()
    for filename in os.listdir(temp_dir):
        if filename.endswith(('.mp4', '.avi', '.mov')):
            file_path = os.path.join(temp_dir, filename)
            try:
                os.unlink(file_path)
            except PermissionError:
                pass

# Main application logic
def main():
    uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

    if uploaded_file is not None:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        if file_extension in ['.jpg', '.jpeg', '.png']:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Detect and Recognize License Plate"):
                st.write("Processing...")
                result_image, detected_plates = predict_and_recognize(np.array(image))
                if result_image is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image, caption="Input Image", use_column_width=True)
                    with col2:
                        st.image(result_image, caption="Processed Image", use_column_width=True)
                    
                    st.subheader("Detected License Plates:")
                    for plate_text, confidence in detected_plates:
                        st.write(f"Plate: {plate_text}, Confidence: {confidence*100:.2f}%")
        
        elif file_extension in ['.mp4', '.avi', '.mov']:
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            st.video(temp_file_path)
            
            if st.button("Detect and Recognize License Plates"):
                st.write("Processing... This may take a while.")
                result_path, all_detected_plates = process_video(temp_file_path)
                if result_path:
                    if os.path.exists(result_path) and os.path.getsize(result_path) > 0:
                        with open(result_path, 'rb') as video_file:
                            video_bytes = video_file.read()
                        st.video(video_bytes)
                        
                        st.subheader("Detected License Plates:")
                        for plate_text, confidence in all_detected_plates:
                            st.write(f"Plate: {plate_text}, Confidence: {confidence*100:.2f}%")
                    else:
                        st.error("Failed to process the video. The output file is empty or does not exist.")
                    
                    try:
                        os.unlink(temp_file_path)
                        os.unlink(result_path)
                    except PermissionError:
                        st.warning("Unable to delete temporary files. They will be cleaned up later.")
                else:
                    st.error("Failed to process the video.")
        
        else:
            st.error(f"Unsupported file type: {file_extension}")

if __name__ == "__main__":
    main()
    cleanup_temp_files()