import os
import cv2
import numpy as np
import tensorflow as tf
import google.generativeai as genai
from flask import Flask, request, redirect, url_for, render_template, send_from_directory, flash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from collections import deque
import time 
prompt_responses = {}

class FrameBuffer:
    def __init__(self, max_size=10):
        self.buffer = deque(maxlen=max_size)
    
    def add_frame(self, frame):
        self.buffer.append(frame)
    
    def get_context(self):
        return list(self.buffer)
    
    def discard_oldest(self, num_frames):
        for _ in range(num_frames):
            if self.buffer:
                self.buffer.popleft()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FRAMES_FOLDER'] = 'processed_frames'
app.config['CONTEXT_FRAMES_FOLDER'] = 'context_frames'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}
 
# Create folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FRAMES_FOLDER'], exist_ok=True)
os.makedirs(app.config['CONTEXT_FRAMES_FOLDER'], exist_ok=True)
 
# Load environment variables
load_dotenv()
genai.configure(api_key='AIzaSyCRU31GS3v7eiqXLPR4gAKRigbIB2i_L4E')
 
# Load autoencoder model
model = tf.keras.models.load_model("autoencoder_video_complex.h5")
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
 
# Function to clear the processed frames folder
def clear_processed_frames_folder():
    for filename in os.listdir(app.config['PROCESSED_FRAMES_FOLDER']):
        file_path = os.path.join(app.config['PROCESSED_FRAMES_FOLDER'], filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
 
# Function to preprocess the frame
def preprocess_frame(frame):
    frame = cv2.resize(frame, (128, 128))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = frame.astype('float32') / 255.0
    frame = np.expand_dims(frame, axis=-1)
    frame = np.expand_dims(frame, axis=0)
    return frame
 
# Function to detect anomalies
def detect_anomaly(autoencoder, frame):
    reconstructed = autoencoder.predict(frame)
    mse = np.mean(np.power(frame - reconstructed, 2))
    threshold = 0.0235
    return mse > threshold


def process_video(video_path, output_dir_context, output_dir_anomaly):
    clear_processed_frames_folder()
    cap = cv2.VideoCapture(video_path)
    frame_buffer = FrameBuffer(max_size=8)
    i = 0
    warm_up_frames = 60
    anomaly_count = 0
    max_frames = 15

    while True:
        ret, frame = cap.read()
        if not ret or anomaly_count >= max_frames:
            break
        
        preprocessed_frame = preprocess_frame(frame)
        
        if i > warm_up_frames:
            if detect_anomaly(model, preprocessed_frame):
                try:
                    # Save anomalous frame
                    output_image_path = os.path.join(output_dir_anomaly, f"anomalous_frame_{anomaly_count}.jpg")
                    cv2.imwrite(output_image_path, frame)
                    
                    # Get buffer context
                    context_frames = frame_buffer.get_context()
                    context_frame_paths = [os.path.join(output_dir_context, f"context_frame_{i}.jpg") for i in range(len(context_frames))]
                    for idx, context_frame in enumerate(context_frames):
                        cv2.imwrite(context_frame_paths[idx], context_frame)
                    
                    # Prepare context for description
                    context_descriptions = []
                    for f in context_frames:
                        image_data = cv2.imencode('.jpg', f)[1].tobytes()
                        description = get_gemini_response('Describe this image in a single sentence with less than 15 words', [{"mime_type": "image/jpeg", "data": image_data}])
                        context_descriptions.append(description)
                        time.sleep(2)
                    
                    context_description = " ".join(context_descriptions)
                    modell = genai.GenerativeModel('gemini-1.5-flash')
                    context = modell.generate_content(f"Please summarize the following text in a single sentence:\n\n{context_description}")
                    
                    # Describe anomaly with context
                    anomaly_description = get_gemini_response(f'Keeping in mind the previous events: {context}, describe the possible anomaly in this image in a single sentence', [{"mime_type": "image/jpeg", "data": cv2.imencode('.jpg', frame)[1].tobytes()}])
                    prompt_responses[f'anomalous_frame_{anomaly_count}.jpg'] = anomaly_description
                    
                    anomaly_count += 1
                except Exception as e:
                    print(f'Error processing frame {anomaly_count}: {e}')
                    break  # Exit the loop to prevent further processing on API error
        
        # Add current frame to buffer
        frame_buffer.add_frame(frame)
        if len(frame_buffer.buffer) > 10:
            frame_buffer.discard_oldest(5)
        
        i += 1
    
    cap.release()
    cv2.destroyAllWindows()

# Function to get Gemini AI response
def get_gemini_response(input_text, image_parts):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([input_text, image_parts[0]])
    return response.text
 
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)
            output_dir_context= app.config['CONTEXT_FRAMES_FOLDER']
            output_dir_anomaly = app.config['PROCESSED_FRAMES_FOLDER']
            process_video(video_path, output_dir_context,output_dir_anomaly)
            return redirect(url_for('show_anomalous_frames'))
    return render_template('upload2.html')
 
@app.route('/processed_frames/<filename>')
def processed_frame(filename):
    return send_from_directory(app.config['PROCESSED_FRAMES_FOLDER'], filename)

@app.route('/context_frames/<filename>')
def context_frames(filename):
    return send_from_directory(app.config['CONTEXT_FRAMES_FOLDER'], filename)
  
 
@app.route('/anomalous_frames', methods=['GET', 'POST'])
def show_anomalous_frames():
    frames = os.listdir(app.config['PROCESSED_FRAMES_FOLDER'])
    if not frames:
        return render_template('frames2111.html', frames=None)
    return render_template('frames2111.html', frames=frames[:15])

@app.route('/frame_description/<filename>')
def frame_description(filename):
    frame_desc = prompt_responses[filename]

    return render_template('frame_description.html', filename=filename, description=frame_desc)

if __name__ == '__main__':
    app.run(debug=True)
# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# import google.generativeai as genai
# from flask import Flask, request, redirect, url_for, render_template, send_from_directory, flash
# from werkzeug.utils import secure_filename
# from dotenv import load_dotenv
# from collections import deque
# import time

# prompt_responses = {}

# class FrameBuffer:
#     def __init__(self, max_size=10):
#         self.buffer = deque(maxlen=max_size)
    
#     def add_frame(self, frame):
#         self.buffer.append(frame)
    
#     def get_context(self):
#         return list(self.buffer)
    
#     def discard_oldest(self, num_frames):
#         for _ in range(num_frames):
#             if self.buffer:
#                 self.buffer.popleft()

# # Initialize Flask app
# app = Flask(__name__)
# app.secret_key = 'supersecretkey'
# app.config['UPLOAD_FOLDER'] = 'uploads'
# app.config['PROCESSED_FRAMES_FOLDER'] = 'processed_frames'
# app.config['CONTEXT_FRAMES_FOLDER'] = 'context_frames'
# app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}

# # Create folders if they don't exist
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# os.makedirs(app.config['PROCESSED_FRAMES_FOLDER'], exist_ok=True)
# os.makedirs(app.config['CONTEXT_FRAMES_FOLDER'], exist_ok=True)

# # Load environment variables
# load_dotenv()
# genai.configure(api_key='AIzaSyCRU31GS3v7eiqXLPR4gAKRigbIB2i_L4E')

# # Load autoencoder model
# model = tf.keras.models.load_model("autoencoder_video_complex.h5")

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# # Function to clear the processed frames folder
# def clear_processed_frames_folder():
#     for filename in os.listdir(app.config['PROCESSED_FRAMES_FOLDER']):
#         file_path = os.path.join(app.config['PROCESSED_FRAMES_FOLDER'], filename)
#         if os.path.isfile(file_path) or os.path.islink(file_path):
#             os.unlink(file_path)

# # Function to preprocess the frame
# def preprocess_frame(frame):
#     frame = cv2.resize(frame, (128, 128))
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     frame = frame.astype('float32') / 255.0
#     frame = np.expand_dims(frame, axis=-1)
#     frame = np.expand_dims(frame, axis=0)
#     return frame

# # Function to detect anomalies
# def detect_anomaly(autoencoder, frame):
#     reconstructed = autoencoder.predict(frame)
#     mse = np.mean(np.power(frame - reconstructed, 2))
#     threshold = 0.0235
#     return mse > threshold

# def process_video(video_path, output_dir_context, output_dir_anomaly):
#     clear_processed_frames_folder()
#     cap = cv2.VideoCapture(video_path)
#     frame_buffer = FrameBuffer(max_size=8)
#     i = 0
#     warm_up_frames = 60
#     anomaly_count = 0
#     max_frames = 15

#     while True:
#         ret, frame = cap.read()
#         if not ret or anomaly_count >= max_frames:
#             break
        
#         preprocessed_frame = preprocess_frame(frame)
        
#         if i > warm_up_frames:
#             if detect_anomaly(model, preprocessed_frame):
#                 try:
#                     # Save anomalous frame
#                     output_image_path = os.path.join(output_dir_anomaly, f"anomalous_frame_{anomaly_count}.jpg")
#                     cv2.imwrite(output_image_path, frame)
                    
#                     # Get buffer context
#                     context_frames = frame_buffer.get_context()
#                     context_frame_paths = [os.path.join(output_dir_context, f"context_frame_{i}.jpg") for i in range(len(context_frames))]
#                     for idx, context_frame in enumerate(context_frames):
#                         cv2.imwrite(context_frame_paths[idx], context_frame)
                    
#                     # Prepare context for description
#                     context_descriptions = []
#                     for f in context_frames:
#                         image_data = cv2.imencode('.jpg', f)[1].tobytes()
#                         description = get_gemini_response('Describe this image in a single sentence with less than 15 words', [{"mime_type": "image/jpeg", "data": image_data}])
#                         context_descriptions.append(description)
#                         time.sleep(2)
                    
#                     context_description = " ".join(context_descriptions)
#                     modell = genai.GenerativeModel('gemini-1.5-flash')
#                     context = modell.generate_content(f"Please summarize the following text in a single sentence:\n\n{context_description}")
                    
#                     # Describe anomaly with context
#                     anomaly_description = get_gemini_response(f'Keeping in mind the previous events: {context}, describe the possible anomaly in this image in a single sentence', [{"mime_type": "image/jpeg", "data": cv2.imencode('.jpg', frame)[1].tobytes()}])
#                     prompt_responses[f'anomalous_frame_{anomaly_count}.jpg'] = anomaly_description
                    
#                     anomaly_count += 1
#                 except Exception as e:
#                     print(f'Error processing frame {anomaly_count}: {e}')
#                     break  # Exit the loop to prevent further processing on API error
        
#         # Add current frame to buffer
#         frame_buffer.add_frame(frame)
#         if len(frame_buffer.buffer) > 10:
#             frame_buffer.discard_oldest(5)
        
#         i += 1
    
#     cap.release()
#     cv2.destroyAllWindows()

# # Function to get Gemini AI response
# def get_gemini_response(input_text, image_parts):
#     model = genai.GenerativeModel('gemini-1.5-flash')
#     try:
#         response = model.generate_content([input_text, image_parts[0]])
#         return response.text
#     except Exception as e:
#         print(f'Error in get_gemini_response: {e}')
#         return "Description not available"

# @app.route('/', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             flash('No file part')
#             return redirect(request.url)
#         file = request.files['file']
#         if file.filename == '':
#             flash('No selected file')
#             return redirect(request.url)
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(video_path)
#             output_dir_context = app.config['CONTEXT_FRAMES_FOLDER']
#             output_dir_anomaly = app.config['PROCESSED_FRAMES_FOLDER']
#             process_video(video_path, output_dir_context, output_dir_anomaly)
#             return redirect(url_for('show_anomalous_frames'))
#     return render_template('upload2.html')

# @app.route('/processed_frames/<filename>')
# def processed_frame(filename):
#     return send_from_directory(app.config['PROCESSED_FRAMES_FOLDER'], filename)

# @app.route('/context_frames/<filename>')
# def context_frames(filename):
#     return send_from_directory(app.config['CONTEXT_FRAMES_FOLDER'], filename)

# @app.route('/anomalous_frames', methods=['GET', 'POST'])
# def show_anomalous_frames():
#     frames = os.listdir(app.config['PROCESSED_FRAMES_FOLDER'])
#     if not frames:
#         return render_template('frames2111.html', frames=None)
#     return render_template('frames2111.html', frames=frames[:15])

# @app.route('/frame_description/<filename>')
# def frame_description(filename):
#     frame_desc = prompt_responses.get(filename, "Description not available")
#     return render_template('frame_description.html', filename=filename, description=frame_desc)

# if __name__ == '__main__':
#     app.run(debug=True)
