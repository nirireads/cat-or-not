import numpy as np
import matplotlib.pyplot as plt
from models import model
from utils import load_h5_file, IMAGE_SIZE


# put your h5 file at data/train_catvnoncat.h5
X_train, Y_train = load_h5_file("data/train/train_catvsnoncat.h5", "train_set_x", "train_set_y")
X_test, Y_test   = load_h5_file("data/test/test_catvsnoncat.h5", "test_set_x", "test_set_y")


# --- Train the model ---
res = model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.005, print_cost=True)

# --- Save parameters ---
w = res["w"]
b = res["b"] if "b" in res else None
# model() above returns w/b; if not, you can call optimize separately to get them.

np.savez("model_params.npz", w=w, b=res["b"], image_size=IMAGE_SIZE)

# --- Plot cost curve ---
costs = res["costs"]
plt.plot(np.arange(len(costs)) * 100, costs)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost vs iterations")
plt.savefig("cost_curve.png")
print("Saved model_params.npz and cost_curve.png")
