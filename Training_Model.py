#importing important libraries
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
import tensorflow.keras.callbacks
from sklearn.metrics import accuracy_score
import os
import cv2 
import glob

#defining input image parameters
IMAGE_WIDTH=400
IMAGE_HEIGHT=400
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3

#labeling images as accordingly by their name(as per orientation)
train_data_path = "/home/developers/orientation/Dataset/NewTrain/NewTrain"
filenames = os.listdir(train_data_path)
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == '90':
        categories.append(90)
    elif category == '0':
        categories.append(0)
    else:
        categories.append(-90)
        
#creating a dataframe consisting image and its respective category
df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})
df['category'] = df['category'].astype('str')    

#defining CNN sequential model

model = Sequential()
#layer1
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),strides = 2))
model.add(Dropout(0.25))
#layer2
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),strides = 2))
model.add(Dropout(0.25))
#layer3
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),strides = 2))
model.add(Dropout(0.25))
#layer4
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),strides = 2))
model.add(Dropout(0.25))
#layer5
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),strides = 2))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()
earlystop = EarlyStopping(patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

callbacks = [earlystop, learning_rate_reduction]
train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size=32
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
#loading and augmenting training dataset
train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    train_data_path, 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    train_data_path, 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)
#training model and saving it
epochs=10
history = model.fit_generator(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks
)
model.save("Mynewmodel1_10.h5")

#loading and augmenting test dataset
test_data_path = "/home/developers/orientation/Dataset/NewTest"
test_filenames = os.listdir(test_data_path)
test_df = pd.DataFrame({
    'filename': test_filenames
})
nb_samples = test_df.shape[0]
print(nb_samples)
test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df, 
    test_data_path, 
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=32,
    shuffle = False
    
)
#predicting results for testset data
predict = model.predict_generator(test_generator)
test_df['category'] = predict.argmax(axis = 1)

#checking accuracy on test dataset by comparing predicted labels with true labels 
true_labels = []
for filename in test_filenames:
    tl = filename.split('.')[0]
    if tl == '90':
        true_labels.append(2)
    elif tl == '0':
        true_labels.append(1)
    else:
        true_labels.append(0)
test_df['True_labels'] = true_labels
files = glob.glob (test_data_path + "/*.jpg")
test_df['Images'] = files

#Below commented code save correct oriented images of disoriented predicted test images.
'''
import math
def rotate_image(mat, angle):
    height, width = mat.shape[:2]
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)

    radians = math.radians(angle)
    sin = math.sin(radians)
    cos = math.cos(radians)
    bound_w = int((height * abs(sin)) + (width * abs(cos)))
    bound_h = int((height * abs(cos)) + (width * abs(sin)))
    rotation_mat[0, 2] += ((bound_w / 2) - image_center[0])
    rotation_mat[1, 2] += ((bound_h / 2) - image_center[1])

    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat

x = 0
path = '/home/developers/orientation/new results/corrected'
for a,i in zip( test_df['category'],test_df['Images']):
    if a == 0:
        print(x)
        x = x+1
        img = cv2.imread(i,1)
        img = rotate_image(img,90)
        cv2.imwrite(os.path.join(path  , str(x) +'.jpg'),img)
    elif a == 1:
        print(x)
        x = x+1
        img = cv2.imread(i,1)
        img = rotate_image(img,0)
        cv2.imwrite(os.path.join(path , str(x) +'.jpg'),img)
    elif a ==2:
        print(x)
        x = x+1
        img = cv2.imread(i,1)
        img = rotate_image(img,-90)
        cv2.imwrite(os.path.join(path , str(x) +'.jpg'),img)
'''
#Saving prediction results in different 7 folders accoringly
#m90_p90 means true orientation is minus 90 but predicted as positive 90, and similarly for others

c = 0
w = 0 
z_p90 = 0
z_m90 = 0 
p90_m90 =0
p90_z =0
m90_z =0
m90_p90 = 0
saved_path = "/home/developers/orientation/finalresults/"
for a,b,i in zip( test_df['category'],test_df['True_labels'],test_df['Images']):
    if a==b :
        c = c+1
        path = saved_path + 'correct'
        img = cv2.imread(i,1)
        cv2.imwrite(os.path.join(path , str(c) +'.jpg'),img)
    elif a!=b and b == 0 and a == 1 :
        m90_z = m90_z +1
        w = w + 1
        path = saved_path + 'm90_zero'
        img = cv2.imread(i,1)
        cv2.imwrite(os.path.join(path , str(m90_z) +'.jpg'),img)  
    elif a!=b and b == 0 and a == 2 :
        m90_p90 = m90_p90+1
        w = w + 1
        path = saved_path + 'm90_p90'
        img = cv2.imread(i,1)
        cv2.imwrite(os.path.join(path , str(m90_p90) +'.jpg'),img)
    elif a!=b and b == 1 and a == 0 :
        z_m90 = z_m90 +1
        w = w + 1
        path = saved_path + 'zero_m90'
        img = cv2.imread(i,1)
        cv2.imwrite(os.path.join(path , str(z_m90) +'.jpg'),img) 
    elif a!=b and b == 1 and a == 2 :
        z_p90 = z_p90 +1
        w = w + 1
        path = saved_path + 'zero_p90'
        img = cv2.imread(i,1)
        cv2.imwrite(os.path.join(path , str(z_p90) +'.jpg'),img)
    elif a!=b and b == 2 and a == 0 :
        p90_m90 = p90_m90 +1
        w = w + 1
        path = saved_path + 'p90_m90'
        img = cv2.imread(i,1)
        cv2.imwrite(os.path.join(path , str(p90_m90) +'.jpg'),img) 
    elif a!=b and b == 2 and a == 1 :
        p90_z = p90_z+1
        w = w + 1
        path = saved_path + 'p90_zero'
        img = cv2.imread(i,1)
        cv2.imwrite(os.path.join(path , str(p90_z) +'.jpg'),img)

#printing accuracy on test dataset
accuracy = accuracy_score(test_df['True_labels'],test_df['category'],normalize=True)
print(accuracy)
