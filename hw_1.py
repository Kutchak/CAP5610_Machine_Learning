import pandas as pd

# train_df = pd.read_csv('Titanic/train.csv')
# test_df = pd.read_csv('Titanic/test.csv')
# combine = [train_df, test_df]
# data = pd.DataFrame(combine)
# print(data.head(1))
train_df = pd.read_csv('Titanic/train.csv')
# print(train_df.head(10))
test_df = pd.read_csv('Titanic/test.csv')
# print(test_df.head(10))
# combine = [train_df, test_df]
combine = pd.concat([train_df, test_df], axis=0, sort=False)
combine.to_csv('combined.csv', index=False)
data = pd.read_csv('combined.csv')
print(data.info())

print(data.iloc[:,5].name)
print('\tcount:',data.iloc[:,5].count())
print('\tmean:',data.iloc[:,5].mean())
print('\tstddev:',data.iloc[:,5].std())
print('\tmin:',data.iloc[:,5].min())
print('\t25%:',data.iloc[:,5].quantile(.25))
print('\t50%:',data.iloc[:,5].quantile(.5))
print('\t75%:',data.iloc[:,5].quantile(.75))
print('\tmax:',data.iloc[:,5].max())

print(data.iloc[:,6].name)
print('\tcount:',data.iloc[:,6].count())
print('\tmean:',data.iloc[:,6].mean())
print('\tstddev:',data.iloc[:,6].std())
print('\tmin:',data.iloc[:,6].min())
print('\t25%:',data.iloc[:,6].quantile(.25))
print('\t50%:',data.iloc[:,6].quantile(.5))
print('\t75%:',data.iloc[:,6].quantile(.75))
print('\tmax:',data.iloc[:,6].max())

print(data.iloc[:,7].name)
print('\tcount:',data.iloc[:,7].count())
print('\tmean:',data.iloc[:,7].mean())
print('\tstddev:',data.iloc[:,7].std())
print('\tmin:',data.iloc[:,7].min())
print('\t25%:',data.iloc[:,7].quantile(.25))
print('\t50%:',data.iloc[:,7].quantile(.5))
print('\t75%:',data.iloc[:,7].quantile(.75))
print('\tmax:',data.iloc[:,7].max())

print(data.iloc[:,9].name)
print('\tcount:',data.iloc[:,9].count())
print('\tmean:',data.iloc[:,9].mean())
print('\tstddev:',data.iloc[:,9].std())
print('\tmin:',data.iloc[:,9].min())
print('\t25%:',data.iloc[:,9].quantile(.25))
print('\t50%:',data.iloc[:,9].quantile(.5))
print('\t75%:',data.iloc[:,9].quantile(.75))
print('\tmax:',data.iloc[:,9].max())

print()
print(data.iloc[:,1].name)
print('\tcount:',data.iloc[:,1].count())
print('\tunique:',data.iloc[:,1].nunique())
print('\tfreq:',data.iloc[:,1].value_counts())

print(data.iloc[:,2].name)
print('\tcount:',data.iloc[:,2].count())
print('\tunique:',data.iloc[:,2].nunique())
print('\tfreq:',data.iloc[:,2].value_counts())

print(data.iloc[:,4].name)
print('\tcount:',data.iloc[:,4].count())
print('\tunique:',data.iloc[:,4].nunique())
print('\tfreq:',data.iloc[:,4].value_counts())

print(data.iloc[:,11].name)
print('\tcount:',data.iloc[:,11].count())
print('\tunique:',data.iloc[:,11].nunique())
print('\tfreq:',data.iloc[:,11].value_counts())




