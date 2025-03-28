import tensorflow as tf

print("TensorFlow version:", tf.__version__)

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
print("Raw predictions (logits):", predictions)

softmax_preds = tf.nn.softmax(predictions).numpy()
print("Softmax probabilities:", softmax_preds)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
initial_loss = loss_fn(y_train[:1], predictions).numpy()
print("Initial loss:", initial_loss)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print("Test accuracy:", test_accuracy)

probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

predicted_probabilities = probability_model(x_test[:5])
print("Predicted probabilities for first 5 test samples:")
print(predicted_probabilities)
