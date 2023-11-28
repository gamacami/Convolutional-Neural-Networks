from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Define the model
women_or_men = Sequential()
women_or_men.add(Conv2D(32, (5, 5), input_shape=(64, 64, 3), activation='relu'))
women_or_men.add(MaxPooling2D(pool_size=(2, 2)))
women_or_men.add(Flatten())
women_or_men.add(Dense(units=256, activation='relu'))
women_or_men.add(Dense(units=256, activation='relu'))
women_or_men.add(Dense(units=2, activation='sigmoid'))

women_or_men.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=20,  # Additional rotation
    width_shift_range=0.2,  # Additional width shift
    height_shift_range=0.2  # Additional height shift
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load the data
training_set = train_datagen.flow_from_directory(
    'Trabajo final/Untitled/train',
    target_size=(64, 64),
    batch_size=40,
    class_mode='sparse'
)

testing_set = test_datagen.flow_from_directory(
    'Trabajo final/Untitled/test',
    target_size=(64, 64),
    batch_size=40,
    class_mode='sparse'
)

# Calculate class weights
class_weights = {0: 1.0, 1: 1.0}  # Adjust weights as needed

# Train the model with class weights
historic = women_or_men.fit(
    training_set,
    epochs=100,
    validation_data=testing_set,
    class_weight=class_weights,
    verbose=1
)

# Plot loss
plt.figure(1)
plt.plot(historic.history['loss'])
plt.show()

# Evaluate the model
results = women_or_men.evaluate(testing_set)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# Confusion Matrix
y_true = testing_set.classes
y_pred = women_or_men.predict(testing_set).argmax(axis=1)

conf_matrix = confusion_matrix(y_true, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
