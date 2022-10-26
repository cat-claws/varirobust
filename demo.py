import numpy as np
import matplotlib.pyplot as plt

original_accuracy = np.load('original_accuracy.npy')
attacked_accuracy = np.load('attacked_accuracy.npy')

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')

print(original_accuracy)
ax.scatter(original_accuracy[:, 0], original_accuracy[:, 1], original_accuracy[:, 2])
ax.scatter(attacked_accuracy[:, 0], attacked_accuracy[:, 1], attacked_accuracy[:, 2])
plt.xlabel('num of neighbours')
plt.ylabel('size of delta')
ax.set_zlabel('accuracy')
ax.legend(['original accuracy', 'attacked accuracy'])
plt.show()