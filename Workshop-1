from google.colab import drive
drive.mount('/content/drive')
output-:Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
%ls
output-:drive/  sample_data/
%cd /content/drive/MyDrive/archive
output-:/content/drive/MyDrive/archive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df= pd.read_csv("tested.csv")
df.head(200)


/////DATA-CLEANING//////

df.describe()
df.isna().sum()
418-327
df = df.dropna(subset=['Fare'])
print(df['Fare'].isna().sum())
df['Cabin']=df['Cabin'].fillna(0)
print(df['Cabin'].isna().sum())
df.Age
median_age = df['Age'].median()
df['Age'].fillna(median_age,inplace=True)
print(df['Age'].isna().sum())
df.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df.head(10)
df['Embarked'] = le.fit_transform(df['Embarked'].astype(str))
df.head()
df['Cabin'] = le.fit_transform(df['Cabin'].astype(str))
df.Cabin
df.Cabin.unique()
len(df.Cabin.unique())
df = df.drop(['Name'], axis=1)
df.head()
df = df.drop(['PassengerId'], axis=1)
df.head()
df['FamilySize'] = df['SibSp'] + df['Parch']
df.head()
df = df.drop(['SibSp'], axis=1)
df = df.drop(['Parch'], axis=1)
df['TravelAlone'] = np.where((df['FamilySize']) == 1, 1, 0)
df.head()
df['TravelAlone'] = np.where((df['FamilySize']) == 1, 1, 0)
df.head()
df.isnull().sum()
df.head()
df = df.drop(['Ticket'],axis=1)
df.head()
plt.hist(df['Age'], bins=20) 
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Histogram of Age')
plt.hist(df['Survived'], bins=20, color='red')
plt.xlabel('Survived')
plt.ylabel('frequency')
plt.title('Histogram of Survived Person')
male_deaths =df[(df['Sex']==1) & (df['Survived']==0)]
female_deaths = df[(df['Sex']==0) & (df['Survived'])]
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
ax[0].hist(male_deaths['Age'], bins=20)
ax[0].set_xlabel('Age')
ax[0].set_ylabel('Count')
ax[0].set_title('Male deaths')

ax[1].hist(female_deaths['Age'], bins=20)
ax[1].set_xlabel('Age')
ax[1].set_ylabel('Count')
ax[1].set_title('Female deaths')
survived = df['Survived'].value_counts()
plt.pie(survived, labels=['Died', 'Survived'], autopct='%1.1f%%')
plt.title('Number of Survivors')
import seaborn as sns
sns.pairplot(df, hue='Survived')
X = df.drop(['Survived'], axis=1)
y = df['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test=train_test_split(X, y, test_size=0.3, random_state = 7)
X_train.shape
X_train.head()
y_train.shapefrom sklearn.svm import SVC
from sklearn.metrics import accuracy_score
svm = SVC(kernel='linear', random_state= 7)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
X_test
test_passenger = [[2,1,27,12.8442, 2, 2, 1,1]]
test_prediction = svm.predict(test_passenger)
if test_prediction[0] == 1:
  print("The passenger is predicted to have survived.")
else:
    print("The passenger is predicted to have died.")





