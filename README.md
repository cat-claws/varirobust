# Robustness (Variance-minimizing Training)
This is the official code of paper:<br>
[Towards Certified Probabilistic Robustness with High Accuracy](https://arxiv.org)<br>


## Requirements
- pytorch
- torchvision

## Demo
```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('WebAgg') 

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')

ax.scatter(your_x, your_y, your_z)
plt.xlabel('Your X label')
plt.ylabel('Your Y label')
ax.set_zlabel('Your Title')
ax.legend(['Your 1st legend'])
plt.show()
```
