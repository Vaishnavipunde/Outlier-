# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 15:19:58 2023

@author: sai
"""



import pandas as pd

#loads the 'ethnic.csv' dataset into a DataFrame named df
df=pd.read_csv("C:/2-dataset/ethnic.csv")


#df.dtypes is used to display the data types of columns before conversion.
df.dtypes


#astype() method in Pandas is used to convert the 'Salaries' column to integer and the 'age' column to float.
#The final df.dtypes displays the updated data types after conversion.

df.Salaries=df.Salaries.astype(int)
df.dtypes

df.age=df.age.astype(float)
df.dtypes




############### identify duplicates ###############

#import education dataset
df_new=pd.read_csv("C:/2-dataset/education.csv")

duplicate=df_new.duplicated()  #duplicated() is used to find duplicate
duplicate                    #here no duplicate so output is false



#import mtcars_dup dataset
df_new1=pd.read_csv("C:/2-dataset/mtcars_dup.csv")
duplicate1=df_new1.duplicated()
duplicate1     #here duplicate  are present so output is true



#calculate how many total row have duplicate items
sum(duplicate1) 




#drop duplicate (used for removing row which contain duplicate values)

df_new2=df_new1.drop_duplicates()  #drop_duplicates() used to remove duplicates
df_new2.duplicated()




##############outlier treatment##################

import pandas as pd
import seaborn as sns

df=pd.read_csv("C:/2-dataset/ethnic.csv")

sns.boxplot(df.Salaries) # there are outlier


sns.boxplot(df.age) #there is no outlier



#let us calculate IQR
IQR=df.Salaries.quantile(0.75)-df.Salaries.quantile(0.25)



#sometimes IQR not show in variable explorer ,it is there but it treated as constant
# but if we will try as I,Iqr,iqr then it is showing ,so write in small letter
#I=df.Salaries.quantile(0.75)-df.Salaries.quantile(0.25)


#Calculate lower and upper limits for outliers in 'Salaries'

lower_limit=df.Salaries.quantile(0.25)-1.5*IQR
upper_limit=df.Salaries.quantile(0.75)+1.5*IQR


# in lower limit there is an negative value in variable exlorer so replace it with 0



#Three techniques for handling outliers:
#Trimming: Rows containing outliers are removed using boolean indexing based on the calculated limits
#Replacement Technique: Outliers are replaced with the calculated upper and lower limits.
#Winsorizer: The Winsorizer class from the Feature-engine library. This method replaces outliers with values using the IQR method.

#  1)trimming  
       
import numpy as np
outliers_df=np.where(df.Salaries>upper_limit,True,np.where(df.Salaries<lower_limit,True,False))

df_trimmed=df.loc[~outliers_df]  #drop the values which contain outliers

df.shape  #Out[38]: (310, 13)  original shape
df_trimmed.shape  #Out[39]: (306, 13) after trimming


##########################################################################################

#  2)replacement technique


df=pd.read_csv("C:/2-dataset/ethnic.csv")
df.describe()

#replace outlier with upperlimit and lowerlimit
df_replaced=pd.DataFrame(np.where(df.Salaries>upper_limit,upper_limit,np.where(df.Salaries<lower_limit,lower_limit,df.Salaries)))

sns.boxplot(df_replaced[0])

###################################################################################################

#  3)winsorizer  (best method for removing outliers)
#import winsorizer
from feature_engine.outliers import Winsorizer 

winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['Salaries'])

#fit winsorizer model to Salaries column
df_t=winsor.fit_transform(df[['Salaries']])  

#check outlier before winsorizer
sns.boxplot(df['Salaries']) 

#check outlier after winsorize
sns.boxplot(df_t['Salaries'])   








