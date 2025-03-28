import tensorflow as tf

print("TensorFlow version:", tf.__version__)

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    # Flatten the 28x28 images into a 1D vector
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # A Dense hidden layer with 128 neurons and ReLU activation
    tf.keras.layers.Dense(128, activation='relu'),
    # A Dropout layer to help prevent overfitting
    tf.keras.layers.Dropout(0.2),
    # The output layer with 10 neurons (one for each digit)
    tf.keras.layers.Dense(10)
])

# Test the untrained model with a sample from the training set
predictions = model(x_train[:1]).numpy()
print("Raw predictions (logits):", predictions)

# Convert logits to probabilities using softmax
softmax_preds = tf.nn.softmax(predictions).numpy()
print("Softmax probabilities:", softmax_preds)

# Define the loss function
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
initial_loss = loss_fn(y_train[:1], predictions).numpy()
print("Initial loss:", initial_loss)

# Compile the model with the Adam optimizer and accuracy metric
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# Train the model for 5 epochs
model.fit(x_train, y_train, epochs=5)

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print("Test accuracy:", test_accuracy)

# Create a new model that includes a softmax layer to get probabilities
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

# Predict probabilities for the first 5 images in the test set
predicted_probabilities = probability_model(x_test[:5])
print("Predicted probabilities for first 5 test samples:")
print(predicted_probabilities)
