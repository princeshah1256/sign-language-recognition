#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


# ### Data loading

# In[2]:


data_dir = "Sign_language_data" 
classes = sorted(os.listdir(data_dir))

print(classes)


# ### Data processing

# In[3]:


img_size = 128

X = []
y = []

for label in classes:
    folder = os.path.join(data_dir, label)

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        img = cv2.imread(path)
        img = cv2.resize(img, (img_size, img_size))
        img = img / 255.0

        X.append(img)
        y.append(classes.index(label))

X = np.array(X)
y = np.array(y)


# ### Data split and setting

# In[4]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

y_train = to_categorical(y_train, len(classes))
y_test  = to_categorical(y_test, len(classes))


# ### CNN Model Implementation

# In[5]:


cnn_model = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)),
    MaxPooling2D(2,2),

    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128,(3,3),activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(256,activation='relu'),
    Dropout(0.5),

    Dense(len(classes),activation='softmax')
])


# In[6]:


cnn_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# In[7]:


history_cnn = cnn_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=35,
    batch_size=32
)


# In[8]:


loss, acc = cnn_model.evaluate(X_test,y_test)
print("Test Accuracy:",acc)


# In[9]:


cnn_model.save("sign_language_model_cnn.h5")


# ### RNN Model implementation

# In[10]:


X_train_rnn = X_train.reshape(X_train.shape[0], 128, 128*3)
X_test_rnn  = X_test.reshape(X_test.shape[0], 128, 128*3)


# In[11]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

rnn_model = Sequential([
    LSTM(128, input_shape=(128,384), return_sequences=False),
    Dropout(0.5),
    Dense(128, activation="relu"),
    Dense(len(classes), activation="softmax")
])


# In[12]:


rnn_model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# In[13]:


history_rnn = rnn_model.fit(
    X_train_rnn, y_train,
    validation_data=(X_test_rnn, y_test),
    epochs=35,
    batch_size=32
)


# In[14]:


loss, acc = rnn_model.evaluate(X_test_rnn,y_test)
print("RNN Accuracy:",acc)


# In[15]:


rnn_model.save("sign_language_model_rnn.h5")


# ### Transfer Learning implementation

# In[16]:


from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

base = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(128,128,3)
)

for layer in base.layers:
    layer.trainable = False

x = GlobalAveragePooling2D()(base.output)
x = Dense(128, activation="relu")(x)
output = Dense(len(classes), activation="softmax")(x)

transfer_model = Model(inputs=base.input, outputs=output)


# In[17]:


transfer_model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# In[18]:


history_transfer = transfer_model.fit(
    X_train, y_train,
    validation_data=(X_test,y_test),
    epochs=35,
    batch_size=32
)


# In[19]:


loss, acc = transfer_model.evaluate(X_test,y_test)
print("Transfer Model Accuracy:",acc)


# In[20]:


transfer_model.save("sign_language_model_tr.h5")


# ### Testing using Transfer Model

# In[21]:


img_path = "test_data/test_image1.jpg"

img = cv2.imread(img_path)
img = cv2.resize(img,(128,128))
img = img/255.0
img = np.expand_dims(img,axis=0)

prediction = transfer_model.predict(img)
class_index = np.argmax(prediction)

print("Predicted:", classes[class_index])


# -------- MATCHING IMAGE --------

predicted_label = classes[class_index]
folder_path = os.path.join(data_dir, predicted_label)

best_score = float("inf")
best_img = None

for file in os.listdir(folder_path):

    path = os.path.join(folder_path, file)

    dataset_img = cv2.imread(path)
    dataset_img = cv2.resize(dataset_img,(128,128))
    dataset_img = dataset_img/255.0

    score = np.mean((img[0] - dataset_img)**2)

    if score < best_score:
        best_score = score
        best_img = dataset_img


# show result
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(img[0])
plt.title("Test Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(best_img)
plt.title(f"Matched: {predicted_label}")
plt.axis("off")

plt.show()


# In[22]:


img_path = "test_data/test_image2.jpg"

img = cv2.imread(img_path)
img = cv2.resize(img,(128,128))
img = img/255.0
img = np.expand_dims(img,axis=0)

prediction = transfer_model.predict(img)
class_index = np.argmax(prediction)

print("Predicted:", classes[class_index])


# -------- MATCHING IMAGE --------

predicted_label = classes[class_index]
folder_path = os.path.join(data_dir, predicted_label)

best_score = float("inf")
best_img = None

for file in os.listdir(folder_path):

    path = os.path.join(folder_path, file)

    dataset_img = cv2.imread(path)
    dataset_img = cv2.resize(dataset_img,(128,128))
    dataset_img = dataset_img/255.0

    score = np.mean((img[0] - dataset_img)**2)

    if score < best_score:
        best_score = score
        best_img = dataset_img


# show result
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(img[0])
plt.title("Test Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(best_img)
plt.title(f"Matched: {predicted_label}")
plt.axis("off")

plt.show()


# In[23]:


img_path = "test_data/test_image3.jpg"

img = cv2.imread(img_path)
img = cv2.resize(img,(128,128))
img = img/255.0
img = np.expand_dims(img,axis=0)

prediction = transfer_model.predict(img)
class_index = np.argmax(prediction)

print("Predicted:", classes[class_index])


# -------- MATCHING IMAGE --------

predicted_label = classes[class_index]
folder_path = os.path.join(data_dir, predicted_label)

best_score = float("inf")
best_img = None

for file in os.listdir(folder_path):

    path = os.path.join(folder_path, file)

    dataset_img = cv2.imread(path)
    dataset_img = cv2.resize(dataset_img,(128,128))
    dataset_img = dataset_img/255.0

    score = np.mean((img[0] - dataset_img)**2)

    if score < best_score:
        best_score = score
        best_img = dataset_img


# show result
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(img[0])
plt.title("Test Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(best_img)
plt.title(f"Matched: {predicted_label}")
plt.axis("off")

plt.show()


# In[24]:


img_path = "test_data/test_image4.png"

img = cv2.imread(img_path)
img = cv2.resize(img,(128,128))
img = img/255.0
img = np.expand_dims(img,axis=0)

prediction = transfer_model.predict(img)
class_index = np.argmax(prediction)

print("Predicted:", classes[class_index])


# -------- MATCHING IMAGE --------

predicted_label = classes[class_index]
folder_path = os.path.join(data_dir, predicted_label)

best_score = float("inf")
best_img = None

for file in os.listdir(folder_path):

    path = os.path.join(folder_path, file)

    dataset_img = cv2.imread(path)
    dataset_img = cv2.resize(dataset_img,(128,128))
    dataset_img = dataset_img/255.0

    score = np.mean((img[0] - dataset_img)**2)

    if score < best_score:
        best_score = score
        best_img = dataset_img


# show result
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(img[0])
plt.title("Test Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(best_img)
plt.title(f"Matched: {predicted_label}")
plt.axis("off")

plt.show()


# In[25]:


img_path = "test_data/test_image5.png"

img = cv2.imread(img_path)
img = cv2.resize(img,(128,128))
img = img/255.0
img = np.expand_dims(img,axis=0)

prediction = transfer_model.predict(img)
class_index = np.argmax(prediction)

print("Predicted:", classes[class_index])


# -------- MATCHING IMAGE --------

predicted_label = classes[class_index]
folder_path = os.path.join(data_dir, predicted_label)

best_score = float("inf")
best_img = None

for file in os.listdir(folder_path):

    path = os.path.join(folder_path, file)

    dataset_img = cv2.imread(path)
    dataset_img = cv2.resize(dataset_img,(128,128))
    dataset_img = dataset_img/255.0

    score = np.mean((img[0] - dataset_img)**2)

    if score < best_score:
        best_score = score
        best_img = dataset_img


# show result
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(img[0])
plt.title("Test Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(best_img)
plt.title(f"Matched: {predicted_label}")
plt.axis("off")

plt.show()


# In[26]:


plt.plot(history_transfer.history['accuracy'])
plt.plot(history_transfer.history['val_accuracy'])
plt.legend(['train','val'])
plt.title("Accuracy")
plt.show()


# ### Testing using CNN

# In[27]:


img_path = "test_data/test_image1.jpg"

img = cv2.imread(img_path)
img = cv2.resize(img,(128,128))
img = img/255.0
img = np.expand_dims(img,axis=0)

prediction = cnn_model.predict(img)
class_index = np.argmax(prediction)

print("Predicted:", classes[class_index])


# -------- MATCHING IMAGE --------

predicted_label = classes[class_index]
folder_path = os.path.join(data_dir, predicted_label)

best_score = float("inf")
best_img = None

for file in os.listdir(folder_path):

    path = os.path.join(folder_path, file)

    dataset_img = cv2.imread(path)
    dataset_img = cv2.resize(dataset_img,(128,128))
    dataset_img = dataset_img/255.0

    score = np.mean((img[0] - dataset_img)**2)

    if score < best_score:
        best_score = score
        best_img = dataset_img


# show result
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(img[0])
plt.title("Test Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(best_img)
plt.title(f"Matched: {predicted_label}")
plt.axis("off")

plt.show()


# In[28]:


img_path = "test_data/test_image2.jpg"

img = cv2.imread(img_path)
img = cv2.resize(img,(128,128))
img = img/255.0
img = np.expand_dims(img,axis=0)

prediction = cnn_model.predict(img)
class_index = np.argmax(prediction)

print("Predicted:", classes[class_index])


# -------- MATCHING IMAGE --------

predicted_label = classes[class_index]
folder_path = os.path.join(data_dir, predicted_label)

best_score = float("inf")
best_img = None

for file in os.listdir(folder_path):

    path = os.path.join(folder_path, file)

    dataset_img = cv2.imread(path)
    dataset_img = cv2.resize(dataset_img,(128,128))
    dataset_img = dataset_img/255.0

    score = np.mean((img[0] - dataset_img)**2)

    if score < best_score:
        best_score = score
        best_img = dataset_img


# show result
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(img[0])
plt.title("Test Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(best_img)
plt.title(f"Matched: {predicted_label}")
plt.axis("off")

plt.show()


# In[29]:


img_path = "test_data/test_image3.jpg"

img = cv2.imread(img_path)
img = cv2.resize(img,(128,128))
img = img/255.0
img = np.expand_dims(img,axis=0)

prediction = cnn_model.predict(img)
class_index = np.argmax(prediction)

print("Predicted:", classes[class_index])


# -------- MATCHING IMAGE --------

predicted_label = classes[class_index]
folder_path = os.path.join(data_dir, predicted_label)

best_score = float("inf")
best_img = None

for file in os.listdir(folder_path):

    path = os.path.join(folder_path, file)

    dataset_img = cv2.imread(path)
    dataset_img = cv2.resize(dataset_img,(128,128))
    dataset_img = dataset_img/255.0

    score = np.mean((img[0] - dataset_img)**2)

    if score < best_score:
        best_score = score
        best_img = dataset_img


# show result
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(img[0])
plt.title("Test Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(best_img)
plt.title(f"Matched: {predicted_label}")
plt.axis("off")

plt.show()


# In[30]:


img_path = "test_data/test_image4.png"

img = cv2.imread(img_path)
img = cv2.resize(img,(128,128))
img = img/255.0
img = np.expand_dims(img,axis=0)

prediction = cnn_model.predict(img)
class_index = np.argmax(prediction)

print("Predicted:", classes[class_index])


# -------- MATCHING IMAGE --------

predicted_label = classes[class_index]
folder_path = os.path.join(data_dir, predicted_label)

best_score = float("inf")
best_img = None

for file in os.listdir(folder_path):

    path = os.path.join(folder_path, file)

    dataset_img = cv2.imread(path)
    dataset_img = cv2.resize(dataset_img,(128,128))
    dataset_img = dataset_img/255.0

    score = np.mean((img[0] - dataset_img)**2)

    if score < best_score:
        best_score = score
        best_img = dataset_img


# show result
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(img[0])
plt.title("Test Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(best_img)
plt.title(f"Matched: {predicted_label}")
plt.axis("off")

plt.show()


# In[31]:


img_path = "test_data/test_image5.png"

img = cv2.imread(img_path)
img = cv2.resize(img,(128,128))
img = img/255.0
img = np.expand_dims(img,axis=0)

prediction = cnn_model.predict(img)
class_index = np.argmax(prediction)

print("Predicted:", classes[class_index])


# -------- MATCHING IMAGE --------

predicted_label = classes[class_index]
folder_path = os.path.join(data_dir, predicted_label)

best_score = float("inf")
best_img = None

for file in os.listdir(folder_path):

    path = os.path.join(folder_path, file)

    dataset_img = cv2.imread(path)
    dataset_img = cv2.resize(dataset_img,(128,128))
    dataset_img = dataset_img/255.0

    score = np.mean((img[0] - dataset_img)**2)

    if score < best_score:
        best_score = score
        best_img = dataset_img


# show result
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(img[0])
plt.title("Test Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(best_img)
plt.title(f"Matched: {predicted_label}")
plt.axis("off")

plt.show()


# In[32]:


plt.plot(history_cnn.history['accuracy'])
plt.plot(history_cnn.history['val_accuracy'])
plt.legend(['train','val'])
plt.title("Accuracy")
plt.show()


# ### Testing using RNN

# In[33]:


img_path = "test_data/test_image1.jpg"

# -------- LOAD IMAGE --------
img = cv2.imread(img_path)
img = cv2.resize(img,(128,128))

original_img = img.copy()   # keep original for display + matching

# normalize for model
img = img/255.0

# reshape for RNN model
img_rnn = img.reshape(1,128,384)


# -------- PREDICTION --------
prediction = rnn_model.predict(img_rnn)
class_index = np.argmax(prediction)

predicted_label = classes[class_index]
print("Predicted:", predicted_label)


# -------- MATCHING IMAGE --------
folder_path = os.path.join(data_dir, predicted_label)

best_score = float("inf")
best_img = None

for file in os.listdir(folder_path):

    path = os.path.join(folder_path, file)

    dataset_img = cv2.imread(path)
    dataset_img = cv2.resize(dataset_img,(128,128))
    dataset_img = dataset_img/255.0

    # compare using original image
    score = np.mean((original_img/255.0 - dataset_img)**2)

    if score < best_score:
        best_score = score
        best_img = dataset_img


# -------- DISPLAY --------
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(original_img)
plt.title("Test Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(best_img)
plt.title(f"Matched: {predicted_label}")
plt.axis("off")

plt.show()


# In[34]:


img_path = "test_data/test_image2.jpg"

# -------- LOAD IMAGE --------
img = cv2.imread(img_path)
img = cv2.resize(img,(128,128))

original_img = img.copy()   # keep original for display + matching

# normalize for model
img = img/255.0

# reshape for RNN model
img_rnn = img.reshape(1,128,384)


# -------- PREDICTION --------
prediction = rnn_model.predict(img_rnn)
class_index = np.argmax(prediction)

predicted_label = classes[class_index]
print("Predicted:", predicted_label)


# -------- MATCHING IMAGE --------
folder_path = os.path.join(data_dir, predicted_label)

best_score = float("inf")
best_img = None

for file in os.listdir(folder_path):

    path = os.path.join(folder_path, file)

    dataset_img = cv2.imread(path)
    dataset_img = cv2.resize(dataset_img,(128,128))
    dataset_img = dataset_img/255.0

    # compare using original image
    score = np.mean((original_img/255.0 - dataset_img)**2)

    if score < best_score:
        best_score = score
        best_img = dataset_img


# -------- DISPLAY --------
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(original_img)
plt.title("Test Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(best_img)
plt.title(f"Matched: {predicted_label}")
plt.axis("off")

plt.show()


# In[35]:


img_path = "test_data/test_image3.jpg"

# -------- LOAD IMAGE --------
img = cv2.imread(img_path)
img = cv2.resize(img,(128,128))

original_img = img.copy()   # keep original for display + matching

# normalize for model
img = img/255.0

# reshape for RNN model
img_rnn = img.reshape(1,128,384)


# -------- PREDICTION --------
prediction = rnn_model.predict(img_rnn)
class_index = np.argmax(prediction)

predicted_label = classes[class_index]
print("Predicted:", predicted_label)


# -------- MATCHING IMAGE --------
folder_path = os.path.join(data_dir, predicted_label)

best_score = float("inf")
best_img = None

for file in os.listdir(folder_path):

    path = os.path.join(folder_path, file)

    dataset_img = cv2.imread(path)
    dataset_img = cv2.resize(dataset_img,(128,128))
    dataset_img = dataset_img/255.0

    # compare using original image
    score = np.mean((original_img/255.0 - dataset_img)**2)

    if score < best_score:
        best_score = score
        best_img = dataset_img


# -------- DISPLAY --------
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(original_img)
plt.title("Test Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(best_img)
plt.title(f"Matched: {predicted_label}")
plt.axis("off")

plt.show()


# In[36]:


img_path = "test_data/test_image4.png"

# -------- LOAD IMAGE --------
img = cv2.imread(img_path)
img = cv2.resize(img,(128,128))

original_img = img.copy()   # keep original for display + matching

# normalize for model
img = img/255.0

# reshape for RNN model
img_rnn = img.reshape(1,128,384)


# -------- PREDICTION --------
prediction = rnn_model.predict(img_rnn)
class_index = np.argmax(prediction)

predicted_label = classes[class_index]
print("Predicted:", predicted_label)


# -------- MATCHING IMAGE --------
folder_path = os.path.join(data_dir, predicted_label)

best_score = float("inf")
best_img = None

for file in os.listdir(folder_path):

    path = os.path.join(folder_path, file)

    dataset_img = cv2.imread(path)
    dataset_img = cv2.resize(dataset_img,(128,128))
    dataset_img = dataset_img/255.0

    # compare using original image
    score = np.mean((original_img/255.0 - dataset_img)**2)

    if score < best_score:
        best_score = score
        best_img = dataset_img


# -------- DISPLAY --------
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(original_img)
plt.title("Test Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(best_img)
plt.title(f"Matched: {predicted_label}")
plt.axis("off")

plt.show()


# In[37]:


img_path = "test_data/test_image5.png"

# -------- LOAD IMAGE --------
img = cv2.imread(img_path)
img = cv2.resize(img,(128,128))

original_img = img.copy()   # keep original for display + matching

# normalize for model
img = img/255.0

# reshape for RNN model
img_rnn = img.reshape(1,128,384)


# -------- PREDICTION --------
prediction = rnn_model.predict(img_rnn)
class_index = np.argmax(prediction)

predicted_label = classes[class_index]
print("Predicted:", predicted_label)


# -------- MATCHING IMAGE --------
folder_path = os.path.join(data_dir, predicted_label)

best_score = float("inf")
best_img = None

for file in os.listdir(folder_path):

    path = os.path.join(folder_path, file)

    dataset_img = cv2.imread(path)
    dataset_img = cv2.resize(dataset_img,(128,128))
    dataset_img = dataset_img/255.0

    # compare using original image
    score = np.mean((original_img/255.0 - dataset_img)**2)

    if score < best_score:
        best_score = score
        best_img = dataset_img


# -------- DISPLAY --------
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(original_img)
plt.title("Test Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(best_img)
plt.title(f"Matched: {predicted_label}")
plt.axis("off")

plt.show()


# In[38]:


plt.plot(history_rnn.history['accuracy'])
plt.plot(history_rnn.history['val_accuracy'])
plt.legend(['train','val'])
plt.title("Accuracy")
plt.show()

