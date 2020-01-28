import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="ticks", color_codes=True)

from matplotlib.ticker import MultipleLocator

# we need to combine all of the training and testing data
train_df = pd.read_csv('Titanic/train.csv')
test_df = pd.read_csv('Titanic/test.csv')
combine = pd.concat([train_df, test_df], axis=0, sort=False)
combine.to_csv('combined.csv', index=False)
data = pd.read_csv('combined.csv')

# partition dataframe into dataframe with only pclass 1 members
data_pclass_1 = data[data['Pclass'] ==1]
print ('correlation between Pclass 1 and survived')

# obtain correlation between pclass and survived
print ('\t',data_pclass_1['Pclass'].corr(data_pclass_1['Survived']), '\n')

# partition dataframe into dataframe with only females
data_sex_female = data[data['Sex'] == 'female']

# count number of females that survived/died
print (data_sex_female['Survived'].value_counts(normalize=True)*100)

# create a histogram of survivors and correlate it with age
age_survived = plt.figure('Survived')
data_survived = data[data['Survived']==1]

# create a histogram of survivors and correlate it with age and pclass =1
age_pclass_1 = plt.figure('pclass=1|survived=1')
data_survived_pclass_1 = data_survived[data_survived['Pclass']==1]
data_survived_pclass_1['Age'].hist(bins=25)

# create a histogram of survivors and correlate it with age and pclass =2
age_pclass_2 = plt.figure('pclass=2|survived=1')
data_survived_pclass_2 = data_survived[data_survived['Pclass']==2]
data_survived_pclass_2['Age'].hist(bins=25)

# create a histogram of survivors and correlate it with age and pclass =3
age_pclass_3 = plt.figure('pclass=3|survived=1')
data_survived_pclass_3 = data_survived[data_survived['Pclass']==3]
data_survived_pclass_3['Age'].hist(bins=25)

data['age_survived'] = pd.qcut(data_survived['Age'], q=10)
print(data['age_survived'])
print(data['age_survived'].value_counts())

# create a histogram of victims and correlate it with age
age_died = plt.figure('Died')
data_died = data[data['Survived']==0]
data_died['Age'].hist(bins=25)

# create a histogram of victims and correlate it with age and pclass =1
age_pclass_1 = plt.figure('pclass=1|survived=0')
data_died_pclass_1 = data_died[data_died['Pclass']==1]
data_died_pclass_1['Age'].hist(bins=25)

# create a histogram of victims and correlate it with age and pclass =2
age_pclass_2 = plt.figure('pclass=2|survived=0')
data_died_pclass_2 = data_died[data_died['Pclass']==2]
data_died_pclass_2['Age'].hist(bins=25)

# create a histogram of victims and correlate it with age and pclass =3
age_pclass_3 = plt.figure('pclass=3|survived=0')
data_died_pclass_3 = data_died[data_died['Pclass']==3]
data_died_pclass_3['Age'].hist(bins=25)

data['age_died'] = pd.qcut(data_died['Age'], q=10)
print(data['age_died'])
print(data['age_died'].value_counts())

data_died_embarked_s = data_died[data_died['Embarked']=='S']
ddes = sns.catplot(x='Sex', y='Fare', kind='bar', data=data_died_embarked_s)
ddes.fig.suptitle('Embarked = S|Survived = 0')

data_died_embarked_c = data_died[data_died['Embarked']=='C']
ddec = sns.catplot(x='Sex', y='Fare', kind='bar', data=data_died_embarked_c)
ddec.fig.suptitle('Embarked = C|Survived = 0')

data_died_embarked_q = data_died[data_died['Embarked']=='Q']
ddeq = sns.catplot(x='Sex', y='Fare', kind='bar', data=data_died_embarked_q)
ddeq.fig.suptitle('Embarked = Q|Survived = 0')

data_survived_embarked_s = data_survived[data_survived['Embarked']=='S']
dses = sns.catplot(x='Sex', y='Fare', kind='bar', data=data_survived_embarked_s)
dses.fig.suptitle('Embarked = S|Survived = 1')

data_survived_embarked_c = data_survived[data_survived['Embarked']=='C']
dsec = sns.catplot(x='Sex', y='Fare', kind='bar', data=data_survived_embarked_c)
dsec.fig.suptitle('Embarked = C|Survived = 1')

data_survived_embarked_q = data_survived[data_survived['Embarked']=='Q']
dseq = sns.catplot(x='Sex', y='Fare', kind='bar', data=data_survived_embarked_q)
dseq.fig.suptitle('Embarked = Q|Survived = 1')

plt.show()
