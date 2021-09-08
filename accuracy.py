
def my_categorical_accuracy(y_true:'list', y_pred:'list'):
    y_ = np.argmax( y_true, axis=-1 )
    y = np.argmax(y_pred, axis=-1)
    acc = np.equal(y_, y)
    acc = np.mean(acc)
    return acc
 
 
# accuracy in keras
# return list for each example
# categorical_accuracy input is Tensor in TensorFlow
from keras.metrics import categorical_accuracy
