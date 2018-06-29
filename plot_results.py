import matplotlib.pyplot as plt
import numpy as np


def read_file():
    train_loss = np.array([])
    test_loss = np.array([])
    train_acc = np.array([])
    test_acc = np.array([])
    with open("results.txt", "r") as f:
        lines = f.readlines()
        i = 0
        for l in lines:
            if l == "" or l == "\n":
                return
            if i % 2 == 0:
                # train line
                parts = l.split()
                train_loss = np.append(train_loss, float(parts[4]))
                train_acc = np.append(train_acc, float(parts[7]))
            else:
                # test line
                parts = l.split()
                test_loss = np.append(test_loss, float(parts[4]))
                test_acc = np.append(test_acc, float(parts[7]))
            i += 1
    return train_loss, test_loss, train_acc, test_acc


train_loss, test_loss, train_acc, test_acc = read_file()


fig, axes = plt.subplots(2, 2, figsize=(20,15))

axes[0,0].plot(train_loss, 'y', label="Training Loss")
axes[0,1].plot(test_loss, 'g', label="Validation Loss")
axes[1,0].plot(train_acc, 'b', label="Training Accuracy")
axes[1,1].plot(test_acc, 'r', label="Validation Accuracy")

axes[0,0].set_xlabel("Epochs")
axes[0,0].set_ylabel("Loss")

axes[1,0].set_xlabel("Epochs")
axes[1,0].set_ylabel("Accuracy")

axes[0,1].set_xlabel("Epochs")
axes[0,1].set_ylabel("Loss")

axes[1,1].set_xlabel("Epochs")
axes[1,1].set_ylabel("Accuracy")

axes[0,1].legend()
axes[0,0].legend()
axes[1,1].legend()
axes[1,0].legend()

plt.tight_layout()
plt.show()
