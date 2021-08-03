#!/usr/bin/env python
# coding: utf-8

# In[22]:


from flask import Flask, redirect, url_for, render_template, request, send_from_directory
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from skimage.transform import resize
import tensorflow as tf
import os
import skimage.io


# In[23]:


UPLOAD_FOLDER = '/tmp'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = tf.keras.models.load_model('model/')


# In[24]:


@app.route("/")
def home():
    return render_template("index.html")


# In[25]:


@app.route('/tmp/<name>')
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)


# In[26]:


@app.route("/explanation", methods = ['POST'])
def explanation():
    image = request.files['filename']
    image.save(os.path.join("/tmp/image.jpg"))
    image = skimage.io.imread(fname='/tmp/image.jpg')
    explainer = lime_image.LimeImageExplainer()
    image = resize(image, (150, 150))
    
    explanation = explainer.explain_instance(image, model.predict, top_labels=2, hide_color=0, num_samples=1000)
    
    path1 = "/tmp/figure1.png"
    path2 = "/tmp/figure2.png"

    #remove any old figure1 if exist
    try:
        os.remove(path1)
    except:
        print("File for path1 does not exist")
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
    plt.imsave(path1,mark_boundaries(temp , mask))
    
    #remove any old figure2 if exist
    try:
        os.remove(path2)
    except:
        print("File for path2 does not exist")
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
    plt.imsave(path2,mark_boundaries(temp , mask))
    
    while True:
        if(os.path.isfile(path1) and os.path.isfile(path2)):
            break
    return render_template("explanation.html")


# In[ ]:


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)


# In[ ]:




