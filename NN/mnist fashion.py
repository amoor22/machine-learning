import tensorflow
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# it is possible to divide the whole array since it is a numpy array
train_images = train_images / 255.0
test_images = test_images / 255.0
# flatten opens lists inside of each other [[1][2][3]] ---> [1 2 3]
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)

predictions = model.predict(test_images)
for x in range(5):
    print("prediction:", class_names[np.argmax(predictions[x])], "|", "actual:", class_names[test_labels[x]])
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("prediction: " + class_names[np.argmax(predictions[i])])
    plt.show()
print(test_acc)

# plt.imshow(train_images[7], cmap=plt.cm.binary)
# plt.show()