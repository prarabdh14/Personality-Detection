import os
import pickle
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video(video_path):
    """
    Process the video and extract features for prediction.
    Modify this function according to your model's requirements.
    """
    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        # Example feature extraction (modify according to your model's requirements)
        features = []
        frame_count = 0
        
        while cap.isOpened() and frame_count < 100:  # Process up to 100 frames
            ret, frame = cap.read()
            if not ret:
                break
                
            # Example feature: average pixel intensity
            # Replace with your actual feature extraction logic
            avg_intensity = np.mean(frame)
            features.append(avg_intensity)
            frame_count += 1
            
        cap.release()
        
        # Ensure we have a fixed-length feature vector
        # Pad or truncate features as needed
        features = np.array(features[:100])  # Take first 100 features
        if len(features) < 100:
            features = np.pad(features, (0, 100 - len(features)))
            
        return features.reshape(1, -1)  # Reshape for model input
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
        
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
        
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the video
        features = process_video(filepath)
        if features is None:
            return jsonify({'error': 'Error processing video'}), 500
            
        # Load the model and make prediction
        try:
            with open('models/personality_model.pkl', 'rb') as f:
                model = pickle.load(f)
            prediction = model.predict(features)[0]
            
            # Map prediction to personality type (modify based on your model's output)
            personality_types = {
                0: "Extrovert",
                1: "Introvert"
                # Add more personality types based on your model
            }
            result = personality_types.get(prediction, "Unknown")
            
        except FileNotFoundError:
            return jsonify({'error': 'Model file not found'}), 500
        except Exception as e:
            return jsonify({'error': f'Prediction error: {str(e)}'}), 500
            
        # Clean up the uploaded file
        os.remove(filepath)
        
        return jsonify({
            'prediction': f"The person appears to be: {result}",
            'confidence': "High"  # You can add confidence scores if your model provides them
        })
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 