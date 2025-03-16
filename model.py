import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os

# Set image size and batch size
image_size = (224, 224)  # You can change this depending on your dataset
batch_size = 32

# Create ImageDataGenerators for training and validation
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1./255)

val_dir = 'SkinDisease/test'
train_dir = 'SkinDisease/train'

train_generator = train_datagen.flow_from_directory(train_dir,target_size=image_size,batch_size=batch_size,class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(val_dir,target_size=image_size,batch_size=batch_size,class_mode='categorical')
def build_model():
    model = Sequential()

    # Convolutional Layer 1
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Convolutional Layer 2
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Convolutional Layer 3
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten and Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Dropout to prevent overfitting

    # Output Layer
    model.add(Dense(train_generator.num_classes, activation='softmax'))

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model
model = build_model()

# Train the model
history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // batch_size,
                    epochs=10,  # You can adjust the number of epochs
                    validation_data=validation_generator,
                    validation_steps=validation_generator.samples // batch_size)

# Save the model
model.save('skin_care_model.h5')

# Evaluate the model on the validation data
score = model.evaluate(validation_generator, steps=validation_generator.samples // batch_size)
print(f"Validation loss: {score[0]}, Validation accuracy: {score[1]}")
def load_model():
    return tf.keras.models.load_model('skin_care_model.h5')

def predict_skin_condition(image_path):
    model = load_model()

    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = tf.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_idx = prediction.argmax(axis=-1)[0]
    return class_idx
