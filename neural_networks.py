import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr  # learning rate
        self.weights1 = np.random.randn(input_dim, hidden_dim)
        self.bias1 = np.zeros((1, hidden_dim))
        self.weights2 = np.random.randn(hidden_dim, output_dim)
        self.bias2 = np.zeros((1, output_dim))
        
        # Select activation function
        if activation == 'tanh':
            self.activation = tanh
            self.activation_derivative = tanh_derivative
        elif activation == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        elif activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        else:
            raise ValueError("Unsupported activation function")

        # For visualization
        self.hidden_activations = None
        self.gradients = None

    def forward(self, X):
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.hidden_activations = self.activation(self.z1)
        self.z2 = np.dot(self.hidden_activations, self.weights2) + self.bias2
        output = sigmoid(self.z2)  # Output layer uses sigmoid
        return output

    def backward(self, X, y):
        m = X.shape[0]  # Number of samples
        
        # Compute output layer error
        output = self.forward(X)
        error = output - y
        
        # Compute gradients for output layer
        dz2 = error * sigmoid_derivative(self.z2)
        dw2 = np.dot(self.hidden_activations.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Compute gradients for hidden layer
        dz1 = np.dot(dz2, self.weights2.T) * self.activation_derivative(self.z1)
        dw1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Store gradients for visualization
        self.gradients = (dw1, dw2)

        # Update weights and biases
        self.weights1 -= self.lr * dw1
        self.bias1 -= self.lr * db1
        self.weights2 -= self.lr * dw2
        self.bias2 -= self.lr * db2

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Perform training steps by calling forward and backward functions
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)
        
    hidden_activations = mlp.hidden_activations  # Corrected to use hidden_activations

    # --- Hidden Space Visualization ---
    ax_hidden.set_title(f"Hidden Layer Features (Step {(frame+1) * 10})")
    ax_hidden.scatter(
        hidden_activations[:, 0], hidden_activations[:, 1], hidden_activations[:, 2],
        c=y.ravel(), cmap="bwr", alpha=0.7
    )
    ax_hidden.set_xlabel("Hidden Neuron 1")
    ax_hidden.set_ylabel("Hidden Neuron 2")
    ax_hidden.set_zlabel("Hidden Neuron 3")

    # Hyperplane Visualization
    xx, yy = np.meshgrid(
        np.linspace(-1, 1, 50),
        np.linspace(-1, 1, 50)
    )
    hidden_hyperplane = -(mlp.weights2[0] * xx + mlp.weights2[1] * yy + mlp.bias2[0]) / (mlp.weights2[2] + 1e-5)
    ax_hidden.plot_surface(xx, yy, hidden_hyperplane, alpha=0.3, color="orange")

    # --- Input Space Decision Boundary ---
    ax_input.set_title(f"Input Space Decision Boundary (Step {(frame+1) * 10})")
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    input_space = np.c_[xx.ravel(), yy.ravel()]
    predictions = mlp.forward(input_space)
    zz = predictions.reshape(xx.shape)
    ax_input.contourf(xx, yy, zz, alpha=0.7, cmap="bwr")
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap="bwr", edgecolors="k")

    # --- Gradient Visualization ---
    dw1, dw2 = mlp.gradients  # Get the weight gradients

    # Define the positions of the nodes
    node_positions = {
        'x1': (0, 0),
        'x2': (0, 1),
        'h1': (0.5, 0),
        'h2': (0.5, 0.5),
        'h3': (0.5, 1),
        'y': (1, 0)
    }

    # Plot the nodes as circles
    for node, (x, y) in node_positions.items():
        ax_gradient.add_patch(Circle((x, y), radius=0.05, label=node, color='b', alpha=0.7))

    # Draw edges between nodes, with thickness proportional to the gradient magnitude
    edges = [
        ('x1', 'h1'), ('x2', 'h2'), ('h2', 'h3'), ('h1', 'h3'),  # Input to hidden layer
        ('h1', 'y'), ('h2', 'y'), ('h3', 'y')  # Hidden to output layer
    ]

    for edge in edges:
        node1, node2 = edge
        x1, y1 = node_positions[node1]
        x2, y2 = node_positions[node2]

        # Calculate the gradient magnitude for the edge
        if node1 == 'x1' or node1 == 'x2':
            weight_gradient = np.linalg.norm(dw1)  # For input to hidden layer
        else:
            weight_gradient = np.linalg.norm(dw2)  # For hidden to output layer
        
        # Use weight_gradient to scale the line thickness (line width)
        ax_gradient.plot([x1, x2], [y1, y2], color='k', lw=weight_gradient*100)  # Line thickness scaled by gradient magnitude

    ax_gradient.set_xlim([-0.1, 1.1])
    ax_gradient.set_ylim([-0.1, 1.1])
    ax_gradient.set_title(f"Gradient Visualization (Step {(frame+1) * 10})")
    ax_gradient.set_xlabel("x-axis")
    ax_gradient.set_ylabel("y-axis")

# Visualization function to create animation
def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()


# Run visualization
if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)