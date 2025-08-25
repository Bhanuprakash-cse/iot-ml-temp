import pandas as pd

def generateAI():
 dataset=pd.read_csv('datacsv.csv')
 dataset=dataset.dropna()
 X=dataset.iloc[:,1].values
 X=X.reshape(-1,1)
 Y=dataset.iloc[:,-1].values

 from sklearn.model_selection import train_test_split
 X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

 from sklearn.neighbors import KNeighborsClassifier
 classifier=KNeighborsClassifier(n_neighbors=5)

 ai=KNeighborsClassifier(n_neighbors=5)

 ai.fit(X_train,Y_train)

 import pickle
 pickle.dump(ai,open('model.pkl','wb'))