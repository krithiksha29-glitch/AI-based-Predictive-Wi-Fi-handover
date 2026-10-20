import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load the trained model and test data
model = keras.models.load_model('handover_predictor.keras')
X_test = np.load('X_test_scaled.npy')
y_test = np.load('y_test.npy')

towers = ['AP1', 'AP2', 'AP3', 'AP4']

# Get the model's predictions on all 56 test examples
predictions = model.predict(X_test)          # 56 x 4 confidence scores
predicted_towers = np.argmax(predictions, axis=1)   # pick the highest one per example

# Build the confusion matrix manually
matrix = np.zeros((4, 4), dtype=int)
for actual, predicted in zip(y_test, predicted_towers):
    matrix[actual][predicted] += 1

print("Confusion Matrix")
print("(rows = actual correct tower, columns = what model predicted)")
print()
print("            Predicted:", "  ".join(towers))
for i, row in enumerate(matrix):
    print(f"Actual {towers[i]}:      ", "   ".join(str(x).rjust(3) for x in row))

print()
# Per-tower accuracy
for i, tower in enumerate(towers):
    total = matrix[i].sum()
    correct = matrix[i][i]
    if total > 0:
        print(f"{tower}: {correct}/{total} correct ({100*correct/total:.1f}%)")
    else:
        print(f"{tower}: no test examples")