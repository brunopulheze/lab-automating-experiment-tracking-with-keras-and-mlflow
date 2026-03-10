import numpy as np
import mlflow
import mlflow.tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import reuters

# Hyperparameters
num_words = 10000
batch_size = 64
learning_rate = 0.002
epochs = 15

# Load data
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=num_words)

# Log params
mlflow.log_param("learning_rate", learning_rate)
mlflow.log_param("batch_size", batch_size)

# Vectorize
def vectorize_sequences(sequences, dimension=num_words):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(x_train)
x_test = vectorize_sequences(x_test)

num_classes = int(max(y_train) + 1)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# ✅ Enable autolog BEFORE compile
mlflow.tensorflow.autolog()

# Build model
model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(num_words,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(x_train, y_train,
          epochs=epochs,
          batch_size=batch_size,
          validation_split=0.2)

loss, accuracy = model.evaluate(x_test, y_test)

# Log final metrics and tag
mlflow.log_metric("test_accuracy", accuracy)
mlflow.log_metric("test_loss", loss)
mlflow.set_tag("project", "reuters_classification")

print(f"Test Accuracy: {accuracy:.4f} | Test Loss: {loss:.4f}")