## import required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import data

Data = pd.read_csv('raw_data.csv')
Data = Data[~(Data == 0).any(axis=1)]


X = Data.iloc[:,2:].values   

#remove 0 from rows 
X = X[~(X == 0).any(axis=1)]

#Standardizing the values 
from  sklearn.preprocessing import StandardScaler
SC_X = StandardScaler()
X = SC_X.fit_transform(X)

# Finding right no of cluster- WCSS

wcss=[]

from sklearn.cluster import KMeans


'''

THE FOLLOWING CODE TAKE AROUND 10 MINUTES TO FIT BASED ON COMPUTING POWER

'''

for i in range(1,21):
    cluster = KMeans(n_clusters=i,init='k-means++',random_state=0)
    cluster.fit(X)
    wcss.append(cluster.inertia_)
    

'''
5 was the right no. of cluster we should make here
'''

#Predict cluster for the dataset

cluster = KMeans(n_clusters=5,init='k-means++',random_state=0)
cluster.fit(X)
n_cluster = cluster.fit_predict(X)

unique_values = set(n_cluster)
print(unique_values)    
    
#(Data['engine_speed']> 5500).any()
#(Data['vehicle_speed']> 150).any()

#X=[X,n_cluster]


gear=[]
for i in n_cluster:
    gear.append(i+1)

print(set(gear))

Data['gear']= gear


'''

THE FOLLOWING CODE WILL POP UP YOU MEMORY ERROR IF YOU DON'T HAVE MINUMUM 32 GB OF RAM OR ELSE YOU 
WOULD HAVE TO TAKE LOW SET OF RECORDS TO AVOID MEMORY ERRORS


'''
### evaluating teh cluster performance
from sklearn.metrics import silhouette_score
silhouette_score(X,n_cluster)


'''

Classifying data


'''


from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

y=n_cluster
# dividing X, y into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
 
# training a DescisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
dtree_model = DecisionTreeClassifier(max_depth = 4).fit(X_train, y_train)
dtree_predictions = dtree_model.predict(X_test)
 
'''

Evaluating the model performance

'''

# creating a confusion matrix
cm = confusion_matrix(y_test, dtree_predictions)


#evaluating model performance
from sklearn.metrics import accuracy_score
accuracy_score(y_test,dtree_predictions)


''''

PLOTTING GRAPH FOR CLASSIFICAION


'''

plt.scatter(dtree_predictions,y_test, color = 'blue')
plt.plot(dtree_predictions,y_test, color ='green')
plt.title("Decision Tree Prediction")
plt.xlabel("pred_gear")
plt.ylabel("actual_gear")
plt.show()
