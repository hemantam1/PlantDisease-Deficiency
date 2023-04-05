import os
import warnings
warnings.simplefilter("ignore")
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from flask import Flask, request, render_template
from tensorflow.keras import backend as K
from os import listdir
K.clear_session()

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
im = ''
result = '...'
percentage = '...'
i = 0
imageName = ''
solution = ''
@app.route("/")
def index():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload():
    global im, result, percentage , i , imageName , solution
    target = os.path.join(APP_ROOT, 'static\\')
    print(f'Target : {target}')

    if not os.path.isdir(target):
        os.mkdir(target)
    for imgg in os.listdir(target):
        try:
            imgPath = target + imgg
            os.remove(imgPath)
            print(f'Removed : {imgPath}')
        except Exception as e:
            print(e)
        
    for file in request.files.getlist("file"):
        print(f'File : {file}')
        i += 1
        imageName = str(i) + '.JPG'
        filename = file.filename
        destination = "/".join([target, imageName])
        print(f'Destination : {destination}')
        file.save(destination)
        print('analysing Image')
        try:
            image = os.listdir('static')
            im = destination
            print(f'Analysing Image : {im}')
        except Exception as e:
            print(e)
        result = "Failed to Analyse"
        percentage = "0 %"
        try:
            detect()
            solution = solutions(result)
        except Exception as e:
            print(f'Error While Loading : {e}')  
    return render_template('complete.html', name=result, accuracy=percentage , img = imageName , soln = solution)


def detect():
    global im, result, percentage
    print(f'Image : {im}')
    # resolution
    ht=50
    wd=50
    classNames = ["Pepper__bell___Bacterial_spot", "Pepper__bell___healthy" , "Potato___Early_blight" , "Potato___healthy" ,  "Potato___Late_blight" ,
        "Tomato_Bacterial_spot","Tomato_Early_blight","Tomato_healthy",
                  "Tomato_Late_blight","Tomato_Leaf_Mold","Tomato_Septoria_leaf_spot",
                  "Tomato_Spider_mites_Two_spotted_spider_mite","Tomato__Target_Spot",
                  "Tomato__Tomato_mosaic_virus","Tomato__Tomato_YellowLeaf__Curl_Virus"]
    totClass = len(classNames)
    print(classNames)
    print(totClass)
    mdl = r"model2.h5"
    image = cv2.imread(im)
    orig = image.copy()
    try:
        image = cv2.resize(image, (ht, wd))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
    except Exception as e:
        print("Error Occured : ",e)
    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = load_model(mdl)
    (zero, one,two, three,four,five,six,seven, eight,nine, ten , eleven, twelve , thirteen , fourteen) = model.predict(image)[0]
    prob = [zero, one,two, three,four,five,six,seven, eight,nine, ten , eleven, twelve , thirteen , fourteen]

    maxProb = max(prob)
    maxIndex = prob.index(maxProb)
    label = classNames[maxIndex]
    proba = maxProb
    result = label
    percentage = float("{0:.2f}".format(proba * 100))
    for i in range(0,totClass):
        print(f'{classNames[i]} : {prob[i]}')



def solutions(disease):
    switcher = {
        "Tomato_Bacterial_spot": Tomato_Bacterial_spot ,
        "Tomato_Early_blight": Tomato_Early_blight ,
        "Tomato_healthy": Tomato_healthy , 
        "Tomato_Late_blight" : Tomato_Late_blight,
        "Tomato_Leaf_Mold" : Tomato_Leaf_Mold,
        "Tomato_Septoria_leaf_spot" : Tomato_Septoria_leaf_spot,
        "Tomato_Spider_mites_Two_spotted_spider_mite" : Tomato_Spider_mites_Two_spotted_spider_mite,
        "Tomato__Target_Spot" : Tomato__Target_Spot,
        "Tomato__Tomato_mosaic_virus" : Tomato__Tomato_mosaic_virus,
        "Tomato__Tomato_YellowLeaf__Curl_Virus" : Tomato__Tomato_YellowLeaf__Curl_Virus,
        }
    return switcher.get(disease,"Not Found In The List")
        
if __name__ == "__main__":
    app.run(port=4555, debug=True)