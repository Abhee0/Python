import numpy as np 
import matplotlib.pyplot as plt 
# Sigmoid Function 
def sigmoid(x): 
    return 1 / (1 + np.exp(-x)) 
# Tanh Function 
def tanh(x): 
    return np.tanh(x) 
# ReLU Function 
def relu(x): 
    return np.maximum(0, x) 
 
# Leaky ReLU Function 
def leaky_relu(x, alpha=0.01): 
    return np.where(x > 0, x, alpha * x) 
 
# Generate Data 
x = np.linspace(-10, 10, 400) 
 
# Plot Functions 
plt.figure(figsize=(10, 8)) 
plt.subplot(2, 2, 1) 
plt.plot(x, sigmoid(x), label='Sigmoid', color='blue') 
plt.title('Sigmoid Activation Function') 
plt.grid(True) 
 
plt.subplot(2, 2, 2) 
plt.plot(x, tanh(x), label='Tanh', color='green') 
plt.title('Tanh Activation Function') 
plt.grid(True) 
 
plt.subplot(2, 2, 3) 
plt.plot(x, relu(x), label='ReLU', color='red') 
plt.title('ReLU Activation Function') 
plt.grid(True) 
plt.subplot(2, 2, 4) 
plt.plot(x, leaky_relu(x), label='Leaky ReLU', color='purple') 
plt.title('Leaky ReLU Activation Function') 
plt.grid(True) 
 
plt.tight_layout() 
plt.show()