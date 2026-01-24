# TerraSegment AI - Intelligent Aerial Image Segmentation

An AI-powered platform for real-time semantic segmentation of aerial imagery. Automatically classify terrain into buildings, roads, vegetation, water, and land using deep learning.

A modern, AI-powered web application for semantic segmentation and analysis of aerial imagery using Flask and TensorFlow.

## Features

✨ **Two Powerful Modes:**

1. **AI Segmentation Mode**
   - Upload any aerial image
   - Real-time semantic segmentation using trained U-Net model
   - Automatic terrain composition analysis
   - Side-by-side visualization of original and segmented images
   - Support for 6 terrain classes: Water, Road, Vegetation, Building, Land, Unlabeled

2. **Database Search Mode**
   - Search pre-computed dataset using interactive sliders
   - Find images matching specific terrain composition
   - Browse 72 segmented aerial images
   - Real-time distance calculation for closest matches

🎨 **Beautiful UI**
- Modern glassmorphism design with dark theme
- Smooth animations and transitions
- Fully responsive (mobile, tablet, desktop)
- Real-time visual feedback

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model (First Time Only)
```bash
python train_model.py
```
This will:
- Load and patch images from `Dataset-Segmentation/`
- Train a U-Net semantic segmentation model
- Save the model as `semantic_segmentation_model.h5`
- Takes ~10-15 minutes depending on your hardware

### 3. Run the Application
```bash
python app.py
```

Then open your browser and go to: `http://localhost:5000`

## Project Structure
```
Inferentia/
├── app.py                              # Flask backend
├── train_model.py                      # Model training script
├── requirements.txt                    # Python dependencies
├── New_dataset.csv                     # Pre-computed image statistics
├── semantic_segmentation_model.h5      # Trained model (generated)
├── templates/
│   └── index.html                      # Web UI
├── static/
│   ├── style.css                       # Styling
│   └── script.js                       # Frontend logic
├── Dataset-Segmentation/
│   ├── images/                         # Original aerial images (72 JPG files)
│   ├── masks/                          # Segmentation masks (72 PNG files)
│   └── classes.json                    # Class definitions
├── Semantic_Segmentation_of_Aerial_Images_Hackathon.ipynb  # Training notebook
└── FeaturesV2.ipynb                    # Feature analysis notebook
```

## API Endpoints

### GET /
Returns the main HTML interface

### POST /api/upload-image
Upload an image for AI segmentation
- **Parameter:** `file` (multipart/form-data)
- **Response:** 
```json
{
  "original_image": "base64_string",
  "segmentation_mask": "base64_string",
  "percentages": {
    "water": 12.34,
    "road": 25.67,
    ...
  }
}
```

### POST /api/get-image
Search database by terrain composition
- **Parameters:** 
```json
{
  "water": 30,
  "road": 25,
  "vegetation": 20,
  "buildings": 15,
  "land": 10
}
```
- **Response:** Closest matching image with actual percentages

## How the AI Works

### U-Net Architecture
- **Input:** 256×256 aerial image (3 channels)
- **Output:** Semantic segmentation mask (6 classes)
- **Encoder:** Downsampling with Conv2D + MaxPooling
- **Decoder:** Upsampling with Conv2DTranspose + skip connections
- **Loss:** Dice Loss + Categorical Focal Loss

### Inference Pipeline
1. Image upload → Resize to 256×256
2. Normalization → U-Net prediction
3. Argmax → Get class for each pixel
4. Colorization → RGB visualization
5. Calculate percentages → Display results

## Terrain Classes
- 🔵 **Water** (blue)
- 🟠 **Road** (orange)  
- 🟢 **Vegetation** (green)
- 🔴 **Building** (red)
- 🟡 **Land** (yellow)
- ⚫ **Unlabeled** (gray)

## Performance Tips

- Images are resized to 256×256 for inference
- First prediction takes ~2-3 seconds (model loading)
- Subsequent predictions are faster (~1 second)
- Works best with aerial/satellite imagery

## Technologies Used

- **Backend:** Flask (Python)
- **ML:** TensorFlow, Keras, Segmentation Models
- **Frontend:** HTML5, CSS3, Vanilla JavaScript
- **Image Processing:** OpenCV, PIL
- **Data:** Pandas, NumPy

## Hackathon Highlights

✅ Full ML pipeline (data → training → inference)  
✅ Real-time model serving  
✅ Production-ready web interface  
✅ Scalable API architecture  
✅ Beautiful, responsive UI  
✅ Advanced deep learning (U-Net)  

## Future Enhancements

- [ ] Batch image processing
- [ ] Model quantization for faster inference
- [ ] WebGL visualization of segmentation
- [ ] Export segmentation results
- [ ] Custom model training via web UI
- [ ] GPU acceleration support

## License

This project is part of a college hackathon for semantic segmentation of aerial images.

---

**Created with ❤️ for the Inferentia Hackathon**
