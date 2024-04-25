# coding=utf-8
import sys
import os, shutil
import glob
import re
import numpy as np
import cv2
import easyocr

# Flask utils
from flask import Flask,flash, request, render_template,send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__, static_url_path='')
app.secret_key = os.urandom(24)

app.config['CARTOON_FOLDER'] = 'cartoon_images'
app.config['UPLOAD_FOLDER'] = 'uploads'


@app.route('/uploads/<filename>')
def upload_img(filename):
    
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/cartoon_images/<filename>')
def cartoon_img(filename):
    
    return send_from_directory(app.config['CARTOON_FOLDER'], filename)



def cartoonize_gray(img):
    # Convert the input image to gray scale 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Performing OTSU threshold
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    
    # Specify structure shape and kernel size. 
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
    
    # Applying dilation on the threshold image
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
    
    # Finding contours
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Convert the grayscale image to color.
    im_color = cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR)

    # Looping through the identified contours
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Drawing a rectangle on copied image
        rect = cv2.rectangle(im_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    return im_color

def extract_text(image_path):
    reader = easyocr.Reader(['en']) # especifica el idioma, en este caso inglés
    result = reader.readtext(image_path)
    text = ' '.join([item[1] for item in result])
    return text

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        style = request.form.get('style')
        print(style)
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        
        f.save(file_path)
        file_name=os.path.basename(file_path)
        
        # reading the uploaded image
        img = cv2.imread(file_path)
        if style=="gray":
            cart_fname =file_name + "_gray_cartoon.jpg"
            cartoonized = cartoonize_gray(img)
            cartoon_path = os.path.join(
                basepath, 'cartoon_images', secure_filename(cart_fname))
            fname=os.path.basename(cartoon_path)
            print(fname)
            cv2.imwrite(cartoon_path,cartoonized)
            
            # Extract text from the cartoonized image
            extracted_txt = extract_text(cartoon_path)
            
            return render_template('predict.html',file_name=file_name, cartoon_file=fname, text=extracted_txt)
        else:
            flash('Please select style')
            return render_template('index.html')  
        
    return ""



if __name__ == '__main__':
        app.run(debug=True, host="localhost", port=8080)

