import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# 1. Load data
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# 2. Preprocess
X_train = X_train / 255.0
X_test = X_test / 255.0
X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# 3. Build model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(10, activation='softmax')
])

# 4. Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Train
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# 6. Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# 7. Predict
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test, axis=1)

# 8. Visualize
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.xlabel(f"Pred: {predicted_labels[i]}, True: {true_labels[i]}")
plt.show()

# 9. Save (optional)
model.save('mnist_model.h5')