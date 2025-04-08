# Video Analysis Backend

This is the backend service for the video analysis application. It processes uploaded videos and predicts personality traits using a machine learning model.

## Setup

1. Create a Python virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your trained model:
- Put your trained model file (`personality_model.pkl`) in the `models/` directory
- The model should be compatible with scikit-learn and expect video features as input

## Running the Server

1. Make sure you're in the backend directory
2. Run the Flask application:
```bash
python app.py
```

The server will start on `http://localhost:5000`

## API Endpoints

### POST /predict
Accepts video uploads and returns personality predictions

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: 
  - video: Video file (mp4, avi, or mov)

**Response:**
```json
{
    "prediction": "The person appears to be: [Personality Type]",
    "confidence": "High"
}
```

## Notes

- Maximum video file size is 100MB
- Supported video formats: MP4, AVI, MOV
- The server processes up to 100 frames from each video
- Uploaded videos are automatically deleted after processing 