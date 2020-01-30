import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# we need to combine all of the training and testing data
train_df = pd.read_csv('Titanic/train.csv')
test_df = pd.read_csv('Titanic/test.csv')
combine = pd.concat([train_df, test_df], axis=0, sort=False)
combine.to_csv('combined.csv', index=False)
data = pd.read_csv('combined.csv')

#===================================================================================================
# QUESTION 9
# partition dataframe into dataframe with only pclass 1 members
# and see the correlation between pclass and survival
#===================================================================================================
data_pclass_1 = data[data['Pclass'] ==1]
print ('correlation between Pclass 1 and survived')
print (data_pclass_1['Survived'].value_counts(normalize=True)*100)

#===================================================================================================
# QUESTION 10
# partition dataframe into dataframe with only females
# and see the correlation between sex and survival
#===================================================================================================
data_sex_female = data[data['Sex'] == 'female']
print ('\ncorrelation between sex and survived')
print (data_sex_female['Survived'].value_counts(normalize=True)*100)

#===================================================================================================
# QUESTION 11
# create a histogram of survivors and correlate it with age
#===================================================================================================
age_survived = plt.figure('Survived question 11')
data_survived = data[data['Survived']==1]
data_survived['Age'].hist(bins=50)

# create a histogram of victims and correlate it with age
age_died = plt.figure('Died question 11')
data_died = data[data['Survived']==0]
data_died['Age'].hist(bins=50)

#===================================================================================================
# QUESTION 12
# create a histogram of survivors and correlate it with age and pclass =1
# create a histogram of survivors and correlate it with age and pclass =2
# create a histogram of survivors and correlate it with age and pclass =3
#===================================================================================================
age_pclass_1 = plt.figure('pclass=1|survived=1')
data_survived_pclass_1 = data_survived[data_survived['Pclass']==1]
data_survived_pclass_1['Age'].hist(bins=25)

age_pclass_2 = plt.figure('pclass=2|survived=1')
data_survived_pclass_2 = data_survived[data_survived['Pclass']==2]
data_survived_pclass_2['Age'].hist(bins=25)

age_pclass_3 = plt.figure('pclass=3|survived=1')
data_survived_pclass_3 = data_survived[data_survived['Pclass']==3]
data_survived_pclass_3['Age'].hist(bins=25)

age_pclass_1 = plt.figure('pclass=1|survived=0')
data_died_pclass_1 = data_died[data_died['Pclass']==1]
data_died_pclass_1['Age'].hist(bins=25)

age_pclass_2 = plt.figure('pclass=2|survived=0')
data_died_pclass_2 = data_died[data_died['Pclass']==2]
data_died_pclass_2['Age'].hist(bins=25)

age_pclass_3 = plt.figure('pclass=3|survived=0')
data_died_pclass_3 = data_died[data_died['Pclass']==3]
data_died_pclass_3['Age'].hist(bins=25)

#===================================================================================================
# QUESTION 13
# create a histogram of victims and correlate it with age and pclass =1
# create a histogram of victims and correlate it with age and pclass =2
# create a histogram of victims and correlate it with age and pclass =3
#===================================================================================================

data_died_embarked_s = data_died[data_died['Embarked']=='S']
ddes_table = data_died_embarked_s.pivot_table(index="Sex", values="Fare")
ddes_table.plot(title='Survived = 0 | Embarked = S', kind='bar')

data_died_embarked_c = data_died[data_died['Embarked']=='C']
ddec_table = data_died_embarked_c.pivot_table(index="Sex", values="Fare")
ddec_table.plot(title='Survived = 0 | Embarked = C', kind='bar')

data_died_embarked_q = data_died[data_died['Embarked']=='Q']
ddeq_table = data_died_embarked_q.pivot_table(index="Sex", values="Fare")
ddeq_table.plot(title='Survived = 0 | Embarked = Q', kind='bar')

data_survived_embarked_s = data_survived[data_survived['Embarked']=='S']
dses_table = data_survived_embarked_s.pivot_table(index="Sex", values="Fare")
dses_table.plot(title='Survived = 1 | Embarked = S', kind='bar')

data_survived_embarked_c = data_survived[data_survived['Embarked']=='C']
dsec_table = data_survived_embarked_c.pivot_table(index="Sex", values="Fare")
dsec_table.plot(title='Survived = 1 | Embarked = C', kind='bar')

data_survived_embarked_q = data_survived[data_survived['Embarked']=='Q']
dseq_table = data_survived_embarked_q.pivot_table(index="Sex", values="Fare")
dseq_table.plot(title='Survived = 1 | Embarked = Q', kind='bar')
plt.show()

#===================================================================================================
# QUESTION 14
# find duplicates of the ticket row
#===================================================================================================
print("\nNumber of duplicated ticket values")
print(data['Ticket'].duplicated().value_counts())

#===================================================================================================
# QUESTION 15
# find missing values in cabin feature
#===================================================================================================
print("\nNumber of missing cabin values")
print(data['Cabin'].isna().value_counts())

#===================================================================================================
# QUESTION 16
# convert Sex string category to Gender numerical feature
#===================================================================================================
print("\nNumber of males and females")
Gender = {'male': 0, 'female': 1}
data['Sex'] = data['Sex'].map(Gender)
print(data['Sex'].value_counts())

#===================================================================================================
# QUESTION 17
# convert missing ages to k nearest to find top-k
#===================================================================================================
# knn = KNeighborsClassifier(n_neighbors=3)
# xtrain = train_df.drop('Age', axis=1)
# xtrain = xtrain.drop('Name', axis=1)
# xtrain = xtrain.drop('Ticket', axis=1)
# xtrain = xtrain.drop('Cabin', axis=1)
# xtrain = xtrain.drop('Embarked', axis=1)
# xtrain = xtrain.drop('Sex', axis=1)
# xtrain = xtrain.dropna()
# ytrain = data['Age']
# knn.fit(xtrain, ytrain)
# print(knn.score(train_df['Survived'], data['Age']))

