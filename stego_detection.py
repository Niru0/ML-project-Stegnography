from flask import Flask, render_template, request, redirect, url_for, flash
import cv2
import numpy as np
import pickle

app = Flask(__name__)
app.secret_key = "secret_key"

# Load the pre-trained model
with open('stego_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Function to process the uploaded image and make a prediction
def process_image(image):
    features = extract_features(image)
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            # Read the image
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            
            # Process the image
            result = process_image(img)
            
            # Display the result
            if result == 1:
                flash('Steganographic content detected in the image.')
            else:
                flash('No steganographic content detected.')
            return redirect(request.url)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
