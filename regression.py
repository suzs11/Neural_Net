import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1, 1, 100)
signal = 2  + x +2 * x * x 
noise = np.random.normal(0, 0.1, 100)
y = signal +  noise
plt.plot(signal, 'b')
plt.plot(y,'g')
plt.plot(noise, 'r')
plt.legend(["Without Noise", "with noise", "noise"], loc = 2)
x_train = x[0:80]
y_train = y[0:80]

plt.figure()
degree = 2
x_train = np.column_stack([np.power(x_train, i) for i in range(0, degree)])
model = np.dot(np.dot(np.linalg.inv(np.dot(x_train.transpose(), x_train)), 
        x_train.transpose()), y_train)
plt.plot(x, y,'g')
predicted = np.dot(model, [np.power(x, i) for i in range(0,degree)])
plt.plot(x, predicted, 'r')
plt.legend(["Actual", "Predicted"], loc=2)
plt.show()

train_rmse1 = np.sqrt(np.sum(np.dot(y[0:80]-predicted[0:80],
    y_train - predicted[0:80])))
test_rmse1 = np.sqrt(np.sum(np.dot(y[80:]-predicted[80:],
    y[80:]-predicted[80:])))
print("Train", train_rmse1)
print("------------------")
print("Test", test_rmse1)


