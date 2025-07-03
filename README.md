# Image Classification

A web-based cat vs dog image classifier built with Streamlit and TensorFlow. This application allows users to upload images and get real-time predictions with confidence scores.

## Live Demo

Access the deployed application here: [https://app-deploy-image-classification-uvb3r7dr6oeje9knciplwf.streamlit.app/](https://app-deploy-image-classification-uvb3r7dr6oeje9knciplwf.streamlit.app/)

## Features

- Interactive web interface built with Streamlit
- Real-time image classification for cats and dogs
- Image preprocessing with 96x96 pixel resizing
- Confidence score visualization with progress bars
- Support for JPG, JPEG, and PNG image formats
- Educational information about classified animals

## How It Works

1. Upload an image using the file uploader
2. Click "Classify Image" to process your upload
3. View the prediction results with confidence percentage
4. Learn interesting facts about the classified animal

## Technical Details

- **Framework**: Streamlit for web interface
- **Machine Learning**: TensorFlow/Keras for model inference
- **Image Processing**: PIL (Python Imaging Library)
- **Model Format**: Pickle serialized classifier
- **Input Size**: 96x96 pixels
- **Classes**: Binary classification (Cat vs Dog)

## Project Structure

```
├── app.py                          # Main Streamlit application
├── A01665895_image_classifier.pkl  # Pre-trained model file
└── requirements.txt                # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sntsemilio/Streamlit-deploy-image-classification.git
cd Streamlit-deploy-image-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

The application provides a simple interface where users can:

- Upload images in common formats (JPG, JPEG, PNG)
- Get instant predictions with confidence scores
- View processed results with visual feedback
- Learn about the classified animals

## Model Information

The classifier uses a pre-trained neural network that:
- Processes images at 96x96 pixel resolution
- Applies normalization (pixel values divided by 255)
- Outputs binary classification results
- Provides confidence scores for predictions

## Deployment

This application is deployed on Streamlit Cloud and can be accessed via the live demo link above. The deployment automatically handles:

- Model loading and caching
- Image preprocessing
- Real-time inference
- User interface rendering

## Contributing

Feel free to fork this repository and submit pull requests for improvements or bug fixes.

## Developer

Developed by [@sntsemilio](https://github.com/sntsemilio)

## Topics

- streamlit
- machine-learning
- image-classification
- tensorflow
- computer-vision
- deep-learning
- python
- web-app
- cat-dog-classifier
- deployment
