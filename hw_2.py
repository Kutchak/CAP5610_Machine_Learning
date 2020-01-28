import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

train_df = pd.read_csv('Titanic/train.csv')
test_df = pd.read_csv('Titanic/test.csv')
combine = pd.concat([train_df, test_df], axis=0, sort=False)
combine.to_csv('combined.csv', index=False)
data = pd.read_csv('combined.csv')

data1 = data[data['Pclass'] ==1]
print ('correlation between Pclass 1 and survived')
print ('\t',data1['Pclass'].corr(data1['Survived']), '\n')

data_sex = data[data['Sex'] == 'female']
print (data_sex['Survived'].value_counts(normalize=True)*100)

age_survived = plt.figure('Survived')
data_survived = data[data['Survived']==1]
data_survived['Age'].hist(bins=25)

data['age_survived'] = pd.qcut(data_survived['Age'], q=10)
print(data['age_survived'])
print(data['age_survived'].value_counts())

# ax = plt.subplot()
# ax.xaxis.set_minor_locator(MultipleLocator(1))

age_died = plt.figure('Died')
data_died = data[data['Survived']==0]
data_died['Age'].hist(bins=25)

data['age_died'] = pd.qcut(data_died['Age'], q=10)
print(data['age_died'])
print(data['age_died'].value_counts())
# bx = plt.subplot()
# bx.xaxis.set_minor_locator(MultipleLocator(1))

plt.show()
