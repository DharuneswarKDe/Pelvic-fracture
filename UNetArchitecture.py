import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose
from tensorflow.keras.models import Model

# Define UNet architecture
def unet(input_shape, num_classes):
    inputs = Input(input_shape)

    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(num_classes, 1, activation='softmax')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    return model

# Assuming you have your CT scan images and corresponding masks prepared
import cv2
import os
import numpy as np

# Define directories containing your data
image_dir = "path/to/ct_scan_images/"
mask_dir = "path/to/masks/"

# List files in the directories
image_files = os.listdir(image_dir)
mask_files = os.listdir(mask_dir)

# Sort the file lists to ensure images and masks are aligned
image_files.sort()
mask_files.sort()

# Initialize empty lists to store images and masks
images = []
masks = []

# Load images and masks
for img_file, mask_file in zip(image_files, mask_files):
    # Load image
    img_path = os.path.join(image_dir, img_file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale image
    img = cv2.resize(img, (256, 256))  # Resize if needed
    images.append(img)

    # Load mask
    mask_path = os.path.join(mask_dir, mask_file)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale image
    mask = cv2.resize(mask, (256, 256))  # Resize if needed
    masks.append(mask)

# Convert lists to numpy arrays
images = np.array(images)
masks = np.array(masks)

# Normalize pixel values of images (optional)
images = images / 255.0

# Perform any additional preprocessing steps as required
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create an instance of ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,  # Rotate images by up to 20 degrees
    width_shift_range=0.1,  # Shift images horizontally by up to 10% of the width
    height_shift_range=0.1,  # Shift images vertically by up to 10% of the height
    shear_range=0.1,  # Shear intensity (angle in counter-clockwise direction in degrees)
    zoom_range=0.1,  # Zoom images by up to 10%
    horizontal_flip=True,  # Flip images horizontally
    vertical_flip=False,  # Do not flip images vertically
    fill_mode='nearest'  # Fill in newly created pixels after rotation or shifting
)

# Example of applying data augmentation to images and masks
def apply_augmentation(images, masks, num_augmented_samples):
    augmented_images = []
    augmented_masks = []
    for img, mask in zip(images, masks):
        img = img.reshape((1,) + img.shape + (1,))
        mask = mask.reshape((1,) + mask.shape + (1,))
        i = 0
        for batch in datagen.flow(img, batch_size=1, seed=42):
            augmented_images.append(batch[0])
            augmented_masks.append(mask)
            i += 1
            if i >= num_augmented_samples:
                break
    return np.array(augmented_images), np.array(augmented_masks)

# Example of applying data augmentation to your images and masks
augmented_images, augmented_masks = apply_augmentation(images, masks, num_augmented_samples=5)

# Make sure your data is preprocessed accordingly

# Define input shape and number of classes
input_shape = (256, 256, 1)  # Adjust according to your input image dimensions
num_classes = 3  # Three classes: left innominate bone, right innominate bone, and sacrum

# Instantiate the UNet model
model = unet(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
# Replace X_train and Y_train with your training data and labels
# Replace validation_data with your validation data if available
model.fit(X_train, Y_train, batch_size=8, epochs=10, validation_split=0.2)

# Once the model is trained, you can use it to predict on new images
# Replace X_test with your test images
predictions = model.predict(X_test)

# Post-processing steps may be required based on the output format of predictions
# You may need to convert probabilities to class labels and visualize the results
