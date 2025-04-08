import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import librosa
import time
import pickle
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import speech_recognition as srecog
import moviepy as mp
from transformers import AutoModel, AutoTokenizer
import copy
from torch.optim.lr_scheduler import ReduceLROnPlateau



CSV_FILE = "/Users/prarabdhatrey/Desktop/personality_detection/backend/VPTD_Dataset.csv"
#VIDEO_FOLDER = "/Users/prarabdhatrey/Desktop/personality_detection/backend/cache"
CACHE_FOLDER = "/Users/prarabdhatrey/Desktop/personality_detection/backend/cache"

# Create cache folder if it doesn't exist
os.makedirs(CACHE_FOLDER, exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the OCEAN scores
df = pd.read_csv(CSV_FILE)
print(f"Dataset loaded with {len(df)} samples")

# Initialize text processing models
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
text_model = AutoModel.from_pretrained("xlm-roberta-base").to(device)
text_model.eval()

# Extract frames from video with efficiency improvements
def extract_frames(video_path, num_frames=100):
    """Extract a limited number of frames from video for efficiency"""
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Unable to open video file {video_path}")
            return np.zeros((num_frames, 224, 224, 3), dtype=np.uint8)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            print(f"Warning: No frames in {video_path}")
            return np.zeros((num_frames, 224, 224, 3), dtype=np.uint8)

        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        cap.release()

        # Pad with blank frames if needed
        while len(frames) < num_frames:
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

        return np.array(frames)
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return np.zeros((num_frames, 224, 224, 3), dtype=np.uint8)

# Extract audio and text features with caching
def extract_audio_and_text(video_path, language="ru-RU", sr=16000):
    """Extract audio features and transcript with caching"""
    # Create paths for cached files
    audio_cache_path = os.path.join(CACHE_FOLDER, os.path.basename(video_path).replace('.mp4', '_audio.npy'))
    text_cache_path = os.path.join(CACHE_FOLDER, os.path.basename(video_path).replace('.mp4', '_text.txt'))

    # Check if files are already cached
    if os.path.exists(audio_cache_path) and os.path.exists(text_cache_path):
        audio_features = np.load(audio_cache_path)
        with open(text_cache_path, 'r') as f:
            transcript = f.read()
        return audio_features, transcript

    # If not cached, extract features
    audio_path = os.path.join(CACHE_FOLDER, os.path.basename(video_path).replace('.mp4', '.wav'))

    try:
        # Convert video to audio if not already done
        if not os.path.exists(audio_path):
            video_clip = mp.VideoFileClip(video_path)
            video_clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
            video_clip.close()

        # Extract audio features
        y, sr = librosa.load(audio_path, sr=sr)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-8)  # Normalize with epsilon
        audio_features = np.mean(mfccs, axis=1)

        # Extract transcript
        recognizer = srecog.Recognizer()
        with srecog.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
        try:
            transcript = recognizer.recognize_google(audio_data, language=language)
        except (srecog.UnknownValueError, srecog.RequestError):
            transcript = ""
            print(f"Failed to transcribe {video_path}")

        # Cache the results
        np.save(audio_cache_path, audio_features)
        with open(text_cache_path, 'w') as f:
            f.write(transcript)

        return audio_features, transcript
    except Exception as e:
        print(f"Error extracting audio/text from {video_path}: {e}")
        # Return default values in case of error
        return np.zeros(40), ""

# Extract text features using transformer
def extract_text_features(transcript):
    """Process transcript with transformer model"""
    if not transcript:
        # Return zeros for empty transcript
        return torch.zeros(768, device=device)

    try:
        # Tokenize and process through transformer
        inputs = tokenizer(transcript, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.enable_grad():
            outputs = text_model(**inputs)
        # Average the hidden states to get sentence representation
        features = outputs.last_hidden_state.mean(dim=1).squeeze(0)
        return features
    except Exception as e:
        print(f"Error extracting text features: {e}")
        return torch.zeros(768, device=device)

# ResNet for Feature Extraction
class Resnet_feature_extract(nn.Module):
    def __init__(self):
        super(Resnet_feature_extract, self).__init__()
        base_model = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(base_model.children())[:-1])  # Remove classifier
        self.fc = nn.Linear(512, 128)  # Feature compression layer

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Multimodal MAML model
class MultimodalMAML(nn.Module):
    def __init__(self):
        super(MultimodalMAML, self).__init__()
        self.video_fc = nn.Linear(128, 32)
        self.audio_fc = nn.Linear(40, 32)
        self.text_fc = nn.Linear(768, 32)
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.5)
        # Add an intermediate layer
        self.combined_fc = nn.Linear(96, 32)
        self.final_fc = nn.Linear(32, 5)

    def forward(self, video_features, audio_features, text_features):
        video_out = torch.relu(self.video_fc(video_features))
        audio_out = torch.relu(self.audio_fc(audio_features))
        text_out = torch.relu(self.text_fc(text_features))
        combined = torch.cat((video_out, audio_out, text_out), dim=1)
        combined = torch.relu(self.combined_fc(combined))
        combined = self.dropout(combined)  # Apply dropout
        return self.final_fc(combined)

class MAML:
    def __init__(self, model, lr_inner=0.05, lr_outer=0.001):  # Reduced inner learning rate
        self.model = model
        self.lr_inner = lr_inner
        self.lr_outer = lr_outer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr_outer, weight_decay=1e-5)  # Added weight decay

    def inner_loop(self, video, audio, text, labels):
        # Inner loop update - use MSE loss consistently
        loss_fn = nn.MSELoss()

        # Ensure inputs require gradients
        video = video.detach().requires_grad_()
        audio = audio.detach().requires_grad_()
        text = text.detach().requires_grad_()

        # Create a copy of the model for the inner loop
        inner_model = copy.deepcopy(self.model)

        # Enable gradient calculation for the inner model

        output = inner_model(video, audio, text)
        loss = loss_fn(output, labels)
        # Get gradients while preserving graph structure
        grads = torch.autograd.grad(
        loss, inner_model.parameters(),
        create_graph=True,
        allow_unused=True)

        updated_params = []
        for param, grad in zip(inner_model.parameters(), grads):
            if grad is not None:
                updated_param = param - self.lr_inner * grad
                updated_params.append(updated_param)
            else:
                updated_params.append(param)

        return updated_params # Return updated parameters


    def outer_loop(self, video, audio, text, labels):
        # Outer loop optimization with gradient clipping
        updated_params = self.inner_loop(video, audio, text, labels)

        #updated_model = copy.deepcopy(self.model)
        # Update the outer model's parameters directly using the inner model's state
        for i, (param, updated_param) in enumerate(zip(self.model.parameters(), updated_params)):
            #if i == 0:  # Only for the first forward pass
                #param.data = updated_param
            #else:
            param.data.copy_(updated_param.data)

        output = self.model(video, audio, text)
        loss = nn.MSELoss()(output, labels)

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

# Dataset class for efficient loading
class PersonalityDataset(Dataset):
    def __init__(self, data_df, resnet_feature_extract, cached_features=None):
        self.data = data_df
        self.resnet_feature_extract = resnet_feature_extract

        self.cached_features = cached_features or {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        id_name = row['Id_name']

        # Check if features are already cached
        if id_name in self.cached_features:
            video_feat, audio_feat, text_feat = self.cached_features[id_name]
        else:
            # Extract features
            video_path = os.path.join(VIDEO_FOLDER, f"{id_name}.mp4")

            # Video features
            frames = extract_frames(video_path)
            frames_tensor = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
            with torch.enable_grad():
                frames_tensor = frames_tensor.to(device)
                video_feat = self.resnet_feature_extract(frames_tensor.mean(dim=0).unsqueeze(0)).squeeze(0)

            # Audio and text features
            audio_features, transcript = extract_audio_and_text(video_path)
            audio_feat = torch.tensor(audio_features, dtype=torch.float32).to(device)
            text_feat = extract_text_features(transcript)
            text_feat = torch.tensor(text_feat, dtype=torch.float32).to(device)

            # Cache the features
            self.cached_features[id_name] = (video_feat, audio_feat, text_feat)

        # Get labels (OCEAN scores)
        ocean_scores = torch.tensor(
            row[['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness']].values.astype(np.float32),
            dtype=torch.float32
        ).to(device)

        return video_feat, audio_feat, text_feat, ocean_scores

# Preprocess and cache features
def preprocess_dataset(df, resnet_feature_extract):
    """Preprocess all videos and cache features"""
    cache_path = os.path.join(CACHE_FOLDER, 'all_features.pkl')

    # Check if cache exists
    if os.path.exists(cache_path):
        print("Loading cached features...")
        with open(cache_path, 'rb') as f:
            cached_features = pickle.load(f)
        return cached_features

    print("Preprocessing all videos (this is done only once)...")
    cached_features = {}

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        id_name = row['Id_name']
        video_path = os.path.join(VIDEO_FOLDER, f"{id_name}.mp4")

        # Extract video features
        frames = extract_frames(video_path)
        frames_tensor = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        with torch.enable_grad():
            frames_tensor = frames_tensor.to(device)
            video_feat = resnet_feature_extract(frames_tensor.mean(dim=0).unsqueeze(0)).squeeze(0)

        # Extract audio and text features
        audio_features, transcript = extract_audio_and_text(video_path)
        audio_feat = torch.tensor(audio_features, dtype=torch.float32).to(device)
        text_feat = extract_text_features(transcript)

        # Cache features
        cached_features[id_name] = (video_feat, audio_feat, text_feat)

    # Save to disk
    with open(cache_path, 'wb') as f:
        pickle.dump(cached_features, f)

    print("Preprocessing complete!")
    return cached_features

# Create support-query sets for meta-learning
def create_meta_batches(dataset, batch_size=4, k_shot=2):
    """Create support and query set batches for meta-learning"""
    indices = list(range(len(dataset)))
    support_sets = []
    query_sets = []

    # Shuffle indices
    np.random.shuffle(indices)

    # Create batches
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i+batch_size]
        if len(batch_indices) < batch_size:
            continue

        # For each instance in the batch, create a support-query pair
        for idx in batch_indices:
            # Sample k instances for support set (including the current one)
            available_indices = [i for i in range(len(dataset)) if i != idx]
            support_indices = np.random.choice(available_indices, k_shot-1, replace=False).tolist() + [idx]

            # The remaining instances in the batch form the query set
            query_indices = [i for i in batch_indices if i not in support_indices]

            # If not enough query samples, continue
            if len(query_indices) == 0:
                continue

            # Create support and query sets
            support_samples = [dataset[i] for i in support_indices]
            query_samples = [dataset[i] for i in query_indices]

            # Format as required by MAML
            s_video = torch.stack([s[0] for s in support_samples])
            s_audio = torch.stack([s[1] for s in support_samples])
            s_text = torch.stack([s[2] for s in support_samples])
            s_labels = torch.stack([s[3] for s in support_samples])

            q_video = torch.stack([q[0] for q in query_samples])
            q_audio = torch.stack([q[1] for q in query_samples])
            q_text = torch.stack([q[2] for q in query_samples])
            q_labels = torch.stack([q[3] for q in query_samples])

            support_sets.append((s_video, s_audio, s_text, s_labels))
            query_sets.append((q_video, q_audio, q_text, q_labels))

    return list(zip(support_sets, query_sets))


# Training function
def train_model(train_dataset, val_dataset, resnet_feature_extract, num_epochs=8):
    """Train the model using MAML"""
    # Initialize models
    maml_model = MultimodalMAML().to(device)
    maml = MAML(maml_model, lr_inner=0.1, lr_outer=0.001)

    # Create scheduler for learning rate
    scheduler = ReduceLROnPlateau(
        maml.optimizer, mode='min', factor=0.7,
        patience=1,  # Reduced patience
        min_lr=1e-6, verbose=True)

    # Track losses
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    trait_mse = {trait: 0.0 for trait in ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness']}

    for epoch in range(8):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        epoch_start = time.time()

        # Create meta-batches for this epoch
        meta_batches = create_meta_batches(train_dataset)

        # Training phase
        maml_model.train()
        running_loss = 0.0
        for support_set, query_set in tqdm(meta_batches, desc="Training"):
          s_video, s_audio, s_text, s_labels = support_set
          q_video, q_audio, q_text, q_labels = query_set
          #s_video = s_video.requires_grad_()
          #s_audio = s_audio.requires_grad_()
          #s_text = s_text.requires_grad_()
          loss = maml.outer_loop(s_video, s_audio, s_text, s_labels)
          running_loss += loss

        # Average training loss
        avg_train_loss = running_loss / len(meta_batches) if meta_batches else 0
        train_losses.append(avg_train_loss)

        # Validation phase
        maml_model.eval()
        val_meta_batches = create_meta_batches(val_dataset)
        running_val_loss = 0.0

        with torch.enable_grad():
            for support_set, query_set in tqdm(val_meta_batches, desc="Validation"):
                # Use the inner loop for adaptation
                adapted_params = maml.inner_loop(
                    support_set[0], support_set[1], support_set[2], support_set[3]
                )

                # Create adapted model
                adapted_model = MultimodalMAML().to(device)
                adapted_model.load_state_dict({
                    name: param for name, param in zip(maml_model.state_dict(), adapted_params)
                })

                # Evaluate on query set
                query_output = adapted_model(query_set[0], query_set[1], query_set[2])
                val_loss = nn.MSELoss()(query_output, query_set[3])
                running_val_loss += val_loss.item()

        # Average validation loss
        avg_val_loss = running_val_loss / len(val_meta_batches) if val_meta_batches else 0
        val_losses.append(avg_val_loss)

        # Update learning rate
        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(maml_model.state_dict(), os.path.join(CACHE_FOLDER, 'best_model.pth'))

        # Print epoch stats
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('MAML Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(CACHE_FOLDER, 'loss_plot.png'))
    
    pickle.dump(maml_model,open('model.pkl','wb'))

    return maml_model

# Function to predict OCEAN scores for a new video
def predict_ocean(video_path, maml_model, resnet_feature_extract):
    """Predict OCEAN scores for a new video"""
    maml_model.eval()

    # Extract video features
    frames = extract_frames(video_path)
    frames_tensor = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    with torch.enable_grad():
        frames_tensor = frames_tensor.to(device)
        video_feat = resnet_feature_extract(frames_tensor.mean(dim=0).unsqueeze(0)).squeeze(0)

    # Extract audio and text features
    audio_features, transcript = extract_audio_and_text(video_path)
    audio_feat = torch.tensor(audio_features, dtype=torch.float32).to(device)
    text_feat = extract_text_features(transcript)

    # Predict
    with torch.enable_grad():
        video_feat = video_feat.unsqueeze(0)
        audio_feat = audio_feat.unsqueeze(0)
        text_feat = text_feat.unsqueeze(0)
        prediction = maml_model(video_feat, audio_feat, text_feat)

    return prediction.cpu().detach().numpy()

# Main execution
def main():
    # Initialize feature extractor
    resnet_feature_extract = Resnet_feature_extract().to(device)
    resnet_feature_extract.eval()

    # Preprocess and cache features
    cached_features = preprocess_dataset(df, resnet_feature_extract)

    # Split data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Training set: {len(train_df)} samples, Validation set: {len(val_df)} samples")

    # Create datasets
    train_dataset = PersonalityDataset(train_df, resnet_feature_extract, cached_features)
    val_dataset = PersonalityDataset(val_df, resnet_feature_extract, cached_features)

    # Train model
    maml_model = train_model(train_dataset, val_dataset, resnet_feature_extract, num_epochs=15)

    # Example prediction
    test_video = os.path.join(VIDEO_FOLDER, 'WIN_20250317_22_11_37_Pro.mp4')
    prediction = predict_ocean(test_video, maml_model, resnet_feature_extract)

    print("\nPredicted OCEAN Scores:")
    traits = ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness']
    for trait, score in zip(traits, prediction[0]):
        print(f"{trait}: {score:.4f}")

    
    pickle.dump(maml_model,open('model.pkl','wb'))



if __name__ == "__main__":
    main()
