## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
       # INCLUDE YOUR CODING AND OUTPUT SCREENSHOTS HERE
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```

![image](https://github.com/varshxnx/EXNO-3-DS/assets/122253525/9a2fdab2-3571-49eb-9e48-03bde91d2e8e)


```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```

![image](https://github.com/varshxnx/EXNO-3-DS/assets/122253525/33efd4e7-4871-4766-ae81-cfa9eac8aa94)

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```

![image](https://github.com/varshxnx/EXNO-3-DS/assets/122253525/f00e06ec-db3c-4d45-9549-ae73bb41ce00)

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/varshxnx/EXNO-3-DS/assets/122253525/b8fce9df-fa97-4d6f-8449-8b45cc1f1a62)


```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```

![image](https://github.com/varshxnx/EXNO-3-DS/assets/122253525/1b7b8fe3-d5a2-41d7-a2a1-5e5f292313c5)


```
df2=pd.concat([df2,enc],axis=1)
df2
```

![image](https://github.com/varshxnx/EXNO-3-DS/assets/122253525/d7921cc6-281f-445d-90a9-820b59a9989f)


```
pd.get_dummies(df2,columns=["nom_0"])
```

![image](https://github.com/varshxnx/EXNO-3-DS/assets/122253525/1d6ad867-8628-4cc5-a3ea-4892c0c34fc1)


```
pip install --upgrade category_encoders
```
![image](https://github.com/varshxnx/EXNO-3-DS/assets/122253525/2e1c490e-0981-4593-adfb-18956ffb847f)


```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```

![image](https://github.com/varshxnx/EXNO-3-DS/assets/122253525/ee1cebec-8d5b-4cdd-a222-558a3063b413)


```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```

![image](https://github.com/varshxnx/EXNO-3-DS/assets/122253525/dd360b68-3350-4f07-b1b1-84f4211880c7)


```
dfb=pd.concat([df,nd],axis=1)
dfb
```

![image](https://github.com/varshxnx/EXNO-3-DS/assets/122253525/e8af9424-ac61-44f6-9d21-269fdbad949b)


```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```

![image](https://github.com/varshxnx/EXNO-3-DS/assets/122253525/3ca55d5a-c2db-478c-8bda-661a073a2109)


```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```

![image](https://github.com/varshxnx/EXNO-3-DS/assets/122253525/34392aa3-a271-46ba-9190-ca3b088bf37c)


```
df.skew()
```

![image](https://github.com/varshxnx/EXNO-3-DS/assets/122253525/84e39763-29a4-485a-9947-de30d8d008fa)


```
np.log(df["Highly Positive Skew"])
```

![image](https://github.com/varshxnx/EXNO-3-DS/assets/122253525/774de509-089e-4aea-aabd-a9d759fb7540)

```
np.reciprocal(df["Moderate Positive Skew"])
```

![image](https://github.com/varshxnx/EXNO-3-DS/assets/122253525/a63ed262-a94a-4bd0-a0ce-0810a8bead8e)


```
np.sqrt(df["Highly Positive Skew"])
```

![image](https://github.com/varshxnx/EXNO-3-DS/assets/122253525/7bd3bc60-bf67-44d2-a30d-90c8dd00ae5d)


```
np.square(df["Highly Positive Skew"])
```

![image](https://github.com/varshxnx/EXNO-3-DS/assets/122253525/afec3343-33f5-4965-a3c9-9650096abca4)

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

![image](https://github.com/varshxnx/EXNO-3-DS/assets/122253525/e8f61c02-90cf-45c3-bc65-d8e1bd136d8a)


```
df.skew()
```

![image](https://github.com/varshxnx/EXNO-3-DS/assets/122253525/5f9524e4-7634-4d80-bdc3-b384ef56e603)


```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
```

![image](https://github.com/varshxnx/EXNO-3-DS/assets/122253525/c4933a5a-d743-42eb-9632-a54f9453ad50)

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/varshxnx/EXNO-3-DS/assets/122253525/d939a6e6-fab5-4706-88e9-1cc71c990064)


```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

![image](https://github.com/varshxnx/EXNO-3-DS/assets/122253525/97c121a0-449b-4d9d-81f0-6759fff040d1)


```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/varshxnx/EXNO-3-DS/assets/122253525/4389a28a-370b-404a-acff-3b4ed0bf3ae7)

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/varshxnx/EXNO-3-DS/assets/122253525/aad8db36-e33c-4141-b7d6-f8f7a617a7c4)

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```

![image](https://github.com/varshxnx/EXNO-3-DS/assets/122253525/323400ab-241f-4ee1-a732-2be3cada98fd)


# RESULT:
       Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.
       
