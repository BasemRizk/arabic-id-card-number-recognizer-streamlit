from roboflow import Roboflow
rf = Roboflow(api_key="enter your api key")
project = rf.workspace().project("card-detection-a4b9m")
model_robo = project.version(1).model
import cv2
import tensorflow as tf
from PIL import Image
from image_preprocess import preprocessing_image
import numpy as np
from keras.models import load_model
model = tf.keras.models.load_model('narabic.h5')
def predict_ocr(baseImg):

    crop=None
    # infer on a local image
    annotations=(model_robo.predict(baseImg, confidence=30, overlap=30).json())
    if len(annotations['predictions'])>0:
        center_x=int(annotations['predictions'][0]['x'])
        center_y=int(annotations['predictions'][0]['y'])
        w=int(annotations['predictions'][0]['width'])
        h=int(annotations['predictions'][0]['height'])
        x=int(center_x-(w/2))
        y=int(center_y-(h/2))
        crop = baseImg[y:y+h, x:x+w]
        crop = cv2.resize(crop, (1000, 200))
        crop = preprocessing_image(crop)
    else:
        return ('Please retake a good image of your id card '),annotations
        
    rois=[]
    img=crop.copy()

    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray=img

    # Threshold the grayscale image to create a binary image
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Invert the binary image
    inv = cv2.bitwise_not(thresh)
    contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    max_h=0

    # Draw a rectangle around each contour
    for contour in contours:
        # Get the bounding box coordinates of the contour
        x, y, w, h = cv2.boundingRect(contour)
        x -= 10
        y -= 10
        w += 20
        h += 20 
        if h>max_h:
            max_h=h
        if w > 30 and h > 30:
            roi = thresh[y:y+h, x:x+w]
            rois.append(roi)
            # Draw the rectangle on the copy of the original image
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    id=[]
    img_size = (28, 28)
    for crop1 in rois:
        image=Image.fromarray(crop1)
        image=image.convert('L')
        image = image.resize(img_size, Image.ANTIALIAS)
        img = np.array(image)
        pred=model.predict(np.expand_dims(img, axis=0),verbose=0)
        val=(np.argmax(pred,axis=1)[0])
        if crop1.shape[0]<(int(max_h*0.6)):
            val=0
        id.append(val)
    str_list = ''.join(map(str, id))
    str_list=list(str_list)
    ar_nums = []
    for num in str_list:
        num_arabic = num.replace('1', '١').replace('2', '٢').replace('3', '٣').replace('4', '٤').replace('5', '٥').replace('6', '٦').replace('7', '٧').replace('8', '٨').replace('9', '٩').replace('0', '٠')
        ar_nums.append(num_arabic)
    str_list = ''.join(ar_nums)

    return str_list
