from google.colab import drive
drive.mount('/content/drive/')

%cd /content/drive/MyDrive/

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
df = pd.read_csv('./Data_set/housing.data', header=None, sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS','CHAS','NOX','RM',
              'AGE','DIS','RAD','TAX',
              'PTRATIO','B','LSTAT','MEDV']
df.head()

df = df.values

X = df[:,0:13]
y = df[:,13]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

///////Data Standadization////////

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)'

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

TensorFlow Keras Neural Network///////
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()


//////Input layer////////

model.add(Dense(26, input_dim=13, kernel_initializer='normal', activation='relu'))

//////Hidden Layer//////

model.add(Dense(13, kernel_initializer='normal', activation='relu'))
model.add(Dense(5,  kernel_initializer='normal', activation='relu'))

///////OutputLayer///////
model.add(Dense(1,  kernel_initializer='normal'))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())


//////Trainmodel//////

model.fit(X_train, y_train, epochs=100, batch_size=5, verbose=1)


y_pred = model.predict(X_test)


score = model.evaluate(X_test, y_test,verbose=0)
print(score)


model.add(Dense(13, kernel_initializer='normal', activation='relu'))
model.add(Dense(13, kernel_initializer='normal', activation='relu'))
model.add(Dense(13, kernel_initializer='normal', activation='relu'))
model.add(Dense(13, kernel_initializer='normal', activation='relu'))


model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images,train_labels,epochs=10, 
                    validation_data=(test_images, test_labels))
                    
                    


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)



