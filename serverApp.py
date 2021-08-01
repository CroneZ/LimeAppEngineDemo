#!/usr/bin/env python
# coding: utf-8

# In[27]:


from flask import Flask, redirect, url_for, render_template, request
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from skimage.transform import resize
import tensorflow as tf
import os
import cv2


# In[28]:


app = Flask(__name__)


# In[29]:


@app.route("/")
def home():
    return render_template("index.html")


# In[30]:


@app.route("/explanation", methods = ['POST'])
def explanation():
    image = request.files['filename']
    image.save(os.path.join(os.getcwd(),"static","image.jpg"))
    image = cv2.imread('static/image.jpg')
    explainer = lime_image.LimeImageExplainer()
    model = tf.keras.models.load_model('model/')
    image = resize(image, (150, 150))
    
    explanation = explainer.explain_instance(image, model.predict, top_labels=2, hide_color=0, num_samples=1000)
    
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
    plt.imsave(os.path.join(os.getcwd(),"static","figure1.png"),mark_boundaries(temp , mask))
    
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
    plt.imsave(os.path.join(os.getcwd(),"static","figure2.png"),mark_boundaries(temp , mask))
    
    return render_template("explanation.html")


# In[ ]:


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)


# In[ ]:




