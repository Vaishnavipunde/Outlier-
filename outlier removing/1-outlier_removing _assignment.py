# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 09:02:15 2023

@author: sai
"""
#see the outlier by using boxplot.
#we can remove outlier using three methods like winsorizer or replacement or trimming.
# in this three method winsorizer is a good method.

import pandas as pd
import seaborn as sns
import numpy as np


df=pd.read_csv('C:/2-dataset/Boston.csv')
df.dtypes


df.info
df.describe
df.duplicated().sum()
df.isnull().sum()
df.columns
df['indus'].unique




#outlier searching

sns.boxplot(df['age'])

sns.boxplot(df['tax'])

sns.boxplot(df['crim'])  #y

sns.boxplot(df['zn']) #y

sns.boxplot(df['indus'])

sns.boxplot(df['chas'])  #y

sns.boxplot(df['nox'])

sns.boxplot(df['rm'])  #y

sns.boxplot(df['dis'])  #y

sns.boxplot(df['rad'])

sns.boxplot(df['ptratio'])  #y

sns.boxplot(df['black'])  #y

sns.boxplot(df['lstat'])  #y

sns.boxplot(df['medv']) #y




#calculate iqr

IQR=df.crim.quantile(0.75)-df.crim.quantile(0.25)
IQR1=df.zn.quantile(0.75)-df.zn.quantile(0.25)
IQR2=df.chas.quantile(0.75)-df.chas.quantile(0.25)
IQR3=df.rm.quantile(0.75)-df.rm.quantile(0.25)
IQR4=df.dis.quantile(0.75)-df.dis.quantile(0.25)
IQR5=df.ptratio.quantile(0.75)-df.ptratio.quantile(0.25)
IQR6=df.black.quantile(0.75)-df.black.quantile(0.25)
IQR7=df.lstat.quantile(0.75)-df.lstat.quantile(0.25)
IQR8=df.medv.quantile(0.75)-df.medv.quantile(0.25)




#calcutate upper_limit and lower_limit

lower_limit=df.crim.quantile(0.25)-1.5*IQR
upper_limit=df.crim.quantile(0.75)+1.5*IQR


lower_limit=df.zn.quantile(0.25)-1.5*IQR1
upper_limit=df.zn.quantile(0.75)+1.5*IQR1

lower_limit=df.chas.quantile(0.25)-1.5*IQR2
upper_limit=df.chas.quantile(0.75)+1.5*IQR2

lower_limit=df.rm.quantile(0.25)-1.5*IQR3
upper_limit=df.rm.quantile(0.75)+1.5*IQR3

lower_limit=df.dis.quantile(0.25)-1.5*IQR4
upper_limit=df.dis.quantile(0.75)+1.5*IQR4

lower_limit=df.ptratio.quantile(0.25)-1.5*IQR5
upper_limit=df.ptratio.quantile(0.75)+1.5*IQR5

lower_limit=df.black.quantile(0.25)-1.5*IQR6
upper_limit=df.black.quantile(0.75)+1.5*IQR6

lower_limit=df.lstat.quantile(0.25)-1.5*IQR7
upper_limit=df.lstat.quantile(0.75)+1.5*IQR7

lower_limit=df.medv.quantile(0.25)-1.5*IQR8
upper_limit=df.medv.quantile(0.75)+1.5*IQR8


##############################################################
#replacing outlier with upper and lower  limit

#1
df_replaced=pd.DataFrame(np.where(df.crim>upper_limit,upper_limit,np.where(df.crim<lower_limit,lower_limit,df.crim)))
sns.boxplot(df_replaced[0])


#2
df_replaced=pd.DataFrame(np.where(df.zn>upper_limit,upper_limit,np.where(df.crim<lower_limit,lower_limit,df.crim)))
sns.boxplot(df_replaced[0])


#3
df_replaced=pd.DataFrame(np.where(df.chas>upper_limit,upper_limit,np.where(df.crim<lower_limit,lower_limit,df.crim)))
sns.boxplot(df_replaced[0])


#4
df_replaced=pd.DataFrame(np.where(df.rm>upper_limit,upper_limit,np.where(df.crim<lower_limit,lower_limit,df.crim)))
sns.boxplot(df_replaced[0])


#5
df_replaced=pd.DataFrame(np.where(df.dis>upper_limit,upper_limit,np.where(df.crim<lower_limit,lower_limit,df.crim)))
sns.boxplot(df_replaced[0])


#6
df_replaced=pd.DataFrame(np.where(df.ptratio>upper_limit,upper_limit,np.where(df.crim<lower_limit,lower_limit,df.crim)))
sns.boxplot(df_replaced[0])


#7
df_replaced=pd.DataFrame(np.where(df.black>upper_limit,upper_limit,np.where(df.crim<lower_limit,lower_limit,df.crim)))
sns.boxplot(df_replaced[0])


#8
df_replaced=pd.DataFrame(np.where(df.lstat>upper_limit,upper_limit,np.where(df.crim<lower_limit,lower_limit,df.crim)))
sns.boxplot(df_replaced[0])


#9
df_replaced=pd.DataFrame(np.where(df.medv>upper_limit,upper_limit,np.where(df.medv<lower_limit,lower_limit,df.medv)))
sns.boxplot(df_replaced[0])




###################################################################
#trimming outliers

#1
outliers_df=np.where(df.age>upper_limit,True,np.where(df.age<lower_limit,True,False))
df_trimmed=df.loc[~outliers_df]
df.shape  
df_trimmed.shape


#2
outliers_df1=np.where(df.tax>upper_limit,True,np.where(df.tax<lower_limit,True,False))
df_trimmed=df.loc[~outliers_df1]
df.shape  
df_trimmed.shape


#3
outliers_df2=np.where(df.crim>upper_limit,True,np.where(df.crim<lower_limit,True,False))
df_trimmed=df.loc[~outliers_df2]
df.shape  
df_trimmed.shape


#4
outliers_df3=np.where(df.zn>upper_limit,True,np.where(df.zn<lower_limit,True,False))
df_trimmed=df.loc[~outliers_df3]
df.shape  
df_trimmed.shape


#5
outliers_df4=np.where(df.indus>upper_limit,True,np.where(df.indus<lower_limit,True,False))
df_trimmed=df.loc[~outliers_df4]
df.shape  
df_trimmed.shape


#6
outliers_df5=np.where(df.chas>upper_limit,True,np.where(df.chas<lower_limit,True,False))
df_trimmed=df.loc[~outliers_df5]
df.shape  
df_trimmed.shape



#7
outliers_df6=np.where(df.nox>upper_limit,True,np.where(df.nox<lower_limit,True,False))
df_trimmed=df.loc[~outliers_df6]
df.shape  
df_trimmed.shape


#8
outliers_df7=np.where(df.rm>upper_limit,True,np.where(df.rm<lower_limit,True,False))
df_trimmed=df.loc[~outliers_df7]
df.shape  
df_trimmed.shape


#9
outliers_df8=np.where(df.dis>upper_limit,True,np.where(df.dis<lower_limit,True,False))
df_trimmed=df.loc[~outliers_df8]
df.shape  
df_trimmed.shape


#10
outliers_df9=np.where(df.rad>upper_limit,True,np.where(df.rad<lower_limit,True,False))
df_trimmed=df.loc[~outliers_df9]
df.shape  
df_trimmed.shape


#11
outliers_df10=np.where(df.ptratio>upper_limit,True,np.where(df.ptratio<lower_limit,True,False))
df_trimmed=df.loc[~outliers_df10]
df.shape  
df_trimmed.shape


#12
outliers_df11=np.where(df.black>upper_limit,True,np.where(df.black<lower_limit,True,False))
df_trimmed=df.loc[~outliers_df11]
df.shape  
df_trimmed.shape


#13
outliers_df12=np.where(df.lstat>upper_limit,True,np.where(df.lstat<lower_limit,True,False))
df_trimmed=df.loc[~outliers_df12]
df.shape  
df_trimmed.shape


#14
outliers_df13=np.where(df.medv>upper_limit,True,np.where(df.medv<lower_limit,True,False))
df_trimmed13=df.loc[~outliers_df13]
df.shape  
df_trimmed13.shape





############################################################################
#winsorizer

#1
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['age'])

df_t=winsor.fit_transform(df[['age']])

sns.boxplot(df['age'])
sns.boxplot(df_t['age'])


#2
from feature_engine.outliers import Winsorizer
winsor1=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['tax'])

df_t=winsor1.fit_transform(df[['tax']])

sns.boxplot(df['tax'])
sns.boxplot(df_t['tax'])


#3
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['crim'])

df_t=winsor.fit_transform(df[['crim']])

sns.boxplot(df['crim'])
sns.boxplot(df_t['crim'])


#4
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['zn'])

df_t=winsor.fit_transform(df[['zn']])

sns.boxplot(df['zn'])
sns.boxplot(df_t['zn'])


#5
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['indus'])

df_t=winsor.fit_transform(df[['indus']])

sns.boxplot(df['indus'])
sns.boxplot(df_t['indus'])


#6
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['chas'])

df_t=winsor.fit_transform(df[['chas']])

sns.boxplot(df['chas'])
sns.boxplot(df_t['chas'])


#7
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['nox'])

df_t=winsor.fit_transform(df[['nox']])

sns.boxplot(df['nox'])
sns.boxplot(df_t['nox'])


#8
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['rm'])

df_t=winsor.fit_transform(df[['rm']])

sns.boxplot(df['rm'])
sns.boxplot(df_t['rm'])


#9
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['dis'])

df_t=winsor.fit_transform(df[['dis']])

sns.boxplot(df['dis'])
sns.boxplot(df_t['dis'])


#10
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['rad'])

df_t=winsor.fit_transform(df[['rad']])

sns.boxplot(df['rad'])
sns.boxplot(df_t['rad'])


#11
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['ptratio'])

df_t=winsor.fit_transform(df[['ptratio']])

sns.boxplot(df['ptratio'])
sns.boxplot(df_t['ptratio'])


#12
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['black'])

df_t=winsor.fit_transform(df[['black']])

sns.boxplot(df['black'])
sns.boxplot(df_t['black'])


#13
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['lstat'])

df_t=winsor.fit_transform(df[['lstat']])

sns.boxplot(df['lstat'])
sns.boxplot(df_t['lstat'])


#14
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['age'])

df_t=winsor.fit_transform(df[['medv']])

sns.boxplot(df['medv'])
sns.boxplot(df_t['medv'])






















