from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import cv2
import numpy as np
import os
from io import BytesIO
import base64
from PIL import Image
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the CSV data
csv_path = os.path.join(os.path.dirname(__file__), 'New_dataset.csv')
csv_data = pd.read_csv(csv_path)

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'semantic_segmentation_model.h5')
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✓ Model loaded successfully")
except:
    print("⚠ Model not found. Run 'python train_model.py' first to train the model.")
    model = None

def load_image(image_path):
    """Load image and convert to RGB"""
    full_path = os.path.join(os.path.dirname(__file__), image_path)
    image = cv2.imread(full_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at path: {full_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def find_closest_image(percentages):
    """Find the closest matching image based on slider values"""
    temp_data = csv_data.copy()
    temp_data['distance'] = (abs(temp_data['Water'] - percentages['water']) + 
                            abs(temp_data['Road'] - percentages['road']) + 
                            abs(temp_data['Vegetation'] - percentages['vegetation']) + 
                            abs(temp_data['Building'] - percentages['buildings']) + 
                            abs(temp_data['Land'] - percentages['land']))
    
    closest_row = temp_data.loc[temp_data['distance'].idxmin()]
    return closest_row

def image_to_base64(image_rgb):
    """Convert numpy array to base64 string"""
    pil_image = Image.fromarray(image_rgb)
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    buffered.seek(0)
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    return img_base64

def segment_image(image_rgb):
    """Run semantic segmentation on image"""
    if model is None:
        raise Exception("Model not loaded. Please train the model first.")
    
    # Prepare image for model
    image_resized = cv2.resize(image_rgb, (256, 256))
    image_normalized = image_resized / 255.0
    image_input = np.expand_dims(image_normalized, axis=0)
    
    # Run inference
    prediction = model.predict(image_input, verbose=0)
    segmentation_mask = np.argmax(prediction[0], axis=-1)
    
    return segmentation_mask

def colorize_mask(mask):
    """Convert numeric mask to RGB image"""
    colors = {
        0: (255, 0, 0),        # Building - red
        1: (255, 255, 0),      # Land - yellow
        2: (192, 192, 192),    # Road - gray
        3: (0, 255, 0),        # Vegetation - green
        4: (0, 0, 255),        # Water - blue
        5: (155, 155, 155)     # Unlabeled - gray
    }
    
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in colors.items():
        colored_mask[mask == class_id] = color
    
    return colored_mask

def calculate_percentages(mask):
    """Calculate terrain percentages from segmentation mask"""
    labels = ['Building', 'Land', 'Road', 'Vegetation', 'Water', 'Unlabeled']
    total_pixels = mask.size
    percentages = {}
    
    for class_id, label in enumerate(labels):
        count = np.sum(mask == class_id)
        percentage = (count / total_pixels) * 100
        percentages[label.lower()] = round(percentage, 2)
    
    return percentages

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload-image', methods=['POST'])
def upload_image():
    """Upload and segment image using AI model"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded. Train the model first with: python train_model.py'}), 500
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read image
        img_array = np.frombuffer(file.read(), np.uint8)
        image_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Segment image
        mask = segment_image(image_rgb)
        
        # Get percentages
        percentages = calculate_percentages(mask)
        
        # Colorize mask
        colored_mask = colorize_mask(mask)
        
        # Convert to base64
        original_base64 = image_to_base64(image_rgb)
        mask_base64 = image_to_base64(colored_mask)
        
        return jsonify({
            'original_image': original_base64,
            'segmentation_mask': mask_base64,
            'percentages': percentages,
            'building': percentages.get('building', 0),
            'land': percentages.get('land', 0),
            'road': percentages.get('road', 0),
            'vegetation': percentages.get('vegetation', 0),
            'water': percentages.get('water', 0),
            'unlabeled': percentages.get('unlabeled', 0)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get-image', methods=['POST'])
def get_image():
    """API endpoint to get image based on percentages"""
    try:
        data = request.json
        percentages = {
            'water': data.get('water', 0),
            'road': data.get('road', 0),
            'vegetation': data.get('vegetation', 0),
            'buildings': data.get('buildings', 0),
            'land': data.get('land', 0)
        }
        
        # Check if sum exceeds 100%
        total = sum(percentages.values())
        if total > 100:
            return jsonify({'error': f'Total percentage exceeds 100%! Current: {total}%'}), 400
        
        # Find closest image
        closest_row = find_closest_image(percentages)
        
        # Load image
        mask_path = closest_row['Image_Path']
        image_path = mask_path.replace('masks', 'images').replace('.png', '.jpg')
        
        image_rgb = load_image(image_path)
        img_base64 = image_to_base64(image_rgb)
        
        return jsonify({
            'image': img_base64,
            'water': float(closest_row['Water']),
            'road': float(closest_row['Road']),
            'vegetation': float(closest_row['Vegetation']),
            'buildings': float(closest_row['Building']),
            'land': float(closest_row['Land']),
            'image_name': os.path.basename(mask_path)
        })
    
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
