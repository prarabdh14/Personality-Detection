from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import numpy as np
import torch
from model import extract_frames, extract_audio_and_text, extract_text_features, Resnet_feature_extract, MultimodalMAML

app = Flask(__name__)
CORS(app)

# Get the current directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuration
UPLOAD_FOLDER = os.path.join(CURRENT_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet_model = Resnet_feature_extract().to(device)
personality_model = MultimodalMAML().to(device)

# Load trained model weights
try:
    personality_model.load_state_dict(torch.load('personality_model.pth'))
    personality_model.eval()
except:
    print("Warning: Could not load model weights. Using untrained model.")

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
        frames = extract_frames(filepath)
        audio_features, transcript = extract_audio_and_text(filepath)
        text_features = extract_text_features(transcript)
        
        # Convert to tensors and ensure proper dimensions
        frames_tensor = torch.from_numpy(frames).float().to(device)
        audio_tensor = torch.from_numpy(audio_features).float().to(device)
        text_tensor = text_features.to(device)
        
        # Extract video features
        with torch.no_grad():
            # Process frames through ResNet
            batch_size = frames_tensor.size(0)
            video_features = []
            for i in range(batch_size):
                frame_features = resnet_model(frames_tensor[i].unsqueeze(0))
                video_features.append(frame_features)
            video_features = torch.cat(video_features, dim=0)
            video_features = video_features.mean(dim=0)  # Average over frames
            
            # Make prediction
            prediction = personality_model(video_features, audio_tensor, text_tensor)
        
        # Convert prediction to personality traits
        ocean_traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
        prediction = prediction.squeeze().cpu().numpy()
        
        # Create response
        result = {
            'prediction': 'The person appears to be:',
            'traits': {trait: float(score) for trait, score in zip(ocean_traits, prediction)}
        }
        
        # Clean up
        os.remove(filepath)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
