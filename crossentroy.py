
def my_categorical_crossentropy(y_true:'list', y_pred:'list'):
    '''
    return: list for each example
    '''
    loss = []
    for ii in range(len(y_true)):
        tem_crossentropy_loss = 0
        for jj in range( len(y_true[ii]) ):
            tem_crossentropy_loss +=  (-1 * y_true[ii][jj] * np.log( y_pred[ii][jj] + 1e-10))
        loss.append( tem_crossentropy_loss )
    return loss
 
 
 
# category cross entropy loss in keras
# return list for each example
# categorical_crossentropy input is Tensor in TensorFlow
from keras.losses import categorical_crossentropy
