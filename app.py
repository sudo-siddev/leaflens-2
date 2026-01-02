import os
from flask import Flask, redirect, render_template, request, jsonify, send_from_directory
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from pathlib import Path

disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')

# Initialize ResNet50 model with 39 classes
num_classes = 39
model = models.resnet50(weights=None)  # Don't load pretrained weights
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("trained_model.pth", map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict(file_path):
    try:
        # Load and preprocess the image
        img = Image.open(file_path)
        img = transform(img)

        # Make a prediction
        with torch.no_grad():
            output = model(img.unsqueeze(0))
            predicted_class = torch.argmax(output, dim=1).item()

        return predicted_class

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        raise e

app = Flask(__name__)

# Ensure uploads directory exists
uploads_dir = os.path.join('static', 'uploads')
os.makedirs(uploads_dir, exist_ok=True)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')
    #comment

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        print(file_path)
        pred = predict(file_path)  # Call the predict function here
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        return render_template('submit.html', title=title, desc=description, prevent=prevent,
                               image_url=image_url, pred=pred, sname=supplement_name, simage=supplement_image_url, buy_link=supplement_buy_link)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image=list(supplement_info['supplement image']),
                           supplement_name=list(supplement_info['supplement name']), disease=list(disease_info['disease_name']), buy=list(supplement_info['buy link']))

@app.route('/documentation')
def documentation():
    # Load training results if available
    training_data = None
    try:
        project_root = Path(__file__).parent.parent
        training_summary_path = project_root / 'training_results' / 'training_summary.json'
        if training_summary_path.exists():
            import json
            with open(training_summary_path, 'r') as f:
                training_data = json.load(f)
                # Format values for display
                if 'validation_metrics' in training_data:
                    training_data['validation_metrics']['accuracy_pct'] = f"{training_data['validation_metrics']['accuracy'] * 100:.2f}"
                    training_data['validation_metrics']['f1_macro_str'] = f"{training_data['validation_metrics']['f1_macro']:.4f}"
                    training_data['validation_metrics']['f1_weighted_str'] = f"{training_data['validation_metrics']['f1_weighted']:.4f}"
                if 'test_metrics' in training_data:
                    training_data['test_metrics']['accuracy_pct'] = f"{training_data['test_metrics']['accuracy'] * 100:.2f}"
                    training_data['test_metrics']['f1_macro_str'] = f"{training_data['test_metrics']['f1_macro']:.4f}"
                    training_data['test_metrics']['f1_weighted_str'] = f"{training_data['test_metrics']['f1_weighted']:.4f}"
    except Exception as e:
        print(f"Error loading training data: {e}")
        training_data = None
    
    return render_template('documentation.html', training_data=training_data)

# Route to serve training results images
@app.route('/static/training_results/<path:filename>')
def training_results(filename):
    # Get the project root (parent of App directory)
    project_root = Path(__file__).parent.parent
    training_results_dir = project_root / 'training_results'
    return send_from_directory(str(training_results_dir), filename)

if __name__ == '__main__':
    # For local development
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)
