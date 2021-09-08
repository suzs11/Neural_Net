import numpy as np
import os
import numpy as np 
from PIL import Image


import tensorflow as tf


from tensorflow.keras.utils import to_categorical
from keras import models, layers, regularizers
from keras.optimizers import RMSprop
from keras.datasets import mnist
import matplotlib.pyplot as plt

train_path = './data_800/train_600/'
train_txt = './data_800/train_600.txt'
x_train_savepath = './data_800/x_train.npy'
y_train_savepath = './data_800/y_train.npy'

test_path = './data_800/test_200/'
test_txt = './data_800/test_200.txt'
x_test_savepath = './data_800/x_test.npy'
y_test_savepath = './data_800/y_test.npy'


def generateds(path, txt):
    f = open(txt, 'r')  # 以只读形式打开txt文件
    contents = f.readlines()  # 读取文件中所有行
    f.close()  # 关闭txt文件
    x, y_ = [], []  # 建立空列表
    for content in contents:  # 逐行取出
        value = content.split()  # 以空格分开，图片路径为value[0] , 标签为value[1] , 存入列表
        img_path = path + value[0]  # 拼出图片路径和文件名
        img = Image.open(img_path)  # 读入图片
        img = np.array(img.convert('L'))  # 图片变为8位宽灰度值的np.array格式
        img = img / 255.  # 数据归一化 （实现预处理）
        x.append(img)  # 归一化后的数据，贴到列表x
        y_.append(value[1])  # 标签贴到列表y_
        print('loading : ' + content)  # 打印状态提示

    x = np.array(x)  # 变为np.array格式
    y_ = np.array(y_)  # 变为np.array格式
    y_ = y_.astype(np.int64)  # 变为64位整型
    return x, y_  # 返回输入特征x，返回标签y_


if os.path.exists(x_train_savepath) and os.path.exists(y_train_savepath) and os.path.exists(
        x_test_savepath) and os.path.exists(y_test_savepath):
    print('-------------Load Datasets-----------------')
    x_train_save = np.load(x_train_savepath)
    y_train = np.load(y_train_savepath)
    x_test_save = np.load(x_test_savepath)
    y_test = np.load(y_test_savepath)
    x_train = np.reshape(x_train_save, (len(x_train_save), 28, 28))
    x_test = np.reshape(x_test_save, (len(x_test_save), 28, 28))
else:
    print('-------------Generate Datasets-----------------')
    x_train, y_train = generateds(train_path, train_txt)
    x_test, y_test = generateds(test_path, test_txt)

    print('-------------Save Datasets-----------------')
    x_train_save = np.reshape(x_train, (len(x_train), -1))
    x_test_save = np.reshape(x_test, (len(x_test), -1))
    np.save(x_train_savepath, x_train_save)
    np.save(y_train_savepath, y_train)
    np.save(x_test_savepath, x_test_save)
    np.save(y_test_savepath, y_test)


np.random.shuffle(x_train)
np.random.shuffle(x_test)



####with the fully connected networks#####

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='sigmoid'),
    tf.keras.layers.Dense(84, activation='sigmoid'),
    tf.keras.layers.Dense(42, activation='tanh'),
    tf.keras.layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])
              
checkpoint_save_path = "./checkpoint1/LeNet.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

history = model.fit(x_train, y_train, batch_size=32, epochs=1000,
           validation_data=(x_test, y_test), validation_freq=1)
model.summary()

# print(model.trainable_variables)
file = open('./weights_LeNet.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

score = model.evaluate(x_train, y_train, verbose=1)
print('Train accuracy', score[1])

# 显示训练集和验证集的acc和loss曲线
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('lenet_3l.pdf')
plt.close()
plt.show()

'''
train_images = x_train.reshape((300, 28*28)).astype('float')
test_images = x_test.reshape((100, 28*28)).astype('float')
print(y_train[0])
print("*"*40)
train_labels = to_categorical(y_train, 2)
print(train_labels[0])
test_labels = to_categorical(y_test, 2)
print("标签的具体数值",test_labels[0])

network = models.Sequential()
network.add(layers.Dense(units=128, activation='relu', input_shape=(28*28, ),
                         kernel_regularizer=regularizers.l1(0.0001)))
#network.add(layers.Dropout(0.01))

network.add(layers.Dense(units=30, activation='relu',
                         kernel_regularizer=regularizers.l1(0.0001)))
#network.add(layers.Dropout(0.01))
network.add(layers.Dense(units=2, activation='softmax'))
network.summary()

#神经网络的训练
network.compile(optimizer=RMSprop(learning_rate=0.001),
        loss='categorical_crossentropy',metrics=['accuracy'])

network.fit(train_images, train_labels, epochs=300, batch_size=128, verbose=2)

#print(network.summary())
#来测试一下模型的性能
y_pre = network.predict(test_images[:5])
print(y_pre, "\n", test_labels[:5])
test_loss, test_accuracy = network.evaluate(test_images, test_labels)
print("test_loss", test_loss, "    test_accuracy", test_accuracy)
'''
