import pandas as pd

train_df = pd.read_csv('Titanic/train.csv')
test_df = pd.read_csv('Titanic/test.csv')
combine = pd.concat([train_df, test_df], axis=0, sort=False)
combine.to_csv('combined.csv', index=False)
data = pd.read_csv('combined.csv')
print(data.info())

print(data['Age'].describe())
print(data['SibSp'].describe())
print(data['Parch'].describe())
print(data['Fare'].describe())
print(data['Survived'].describe())
print(data['Pclass'].describe())
print(data['Sex'].describe())
print(data['Embarked'].describe())
