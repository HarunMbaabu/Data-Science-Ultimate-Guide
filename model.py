#import packages 
import seaborn as sns 
import matplotlib.pyplot as plt 
# from sklearn.tree import DecisionTreeClassifier
from sklearn import svm 
from sklearn.model_selection import train_test_split 

dataset = sns.load_dataset("iris") 


X = dataset.drop(columns=["species"])
y = dataset["species"] 


X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.2)


# model  = DecisionTreeClassifier() 
model = svm.SVC(kernel="linear")



model.fit(X, y)
result = model.predict([[4, 5,3,2]]) 

print(result) 

array(['versicolor'], dtype=object)
model.score(X_test, y_test )