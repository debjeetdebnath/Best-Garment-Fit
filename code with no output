import pandas as pd
import numpy as np 

df1 = pd.read_json(r'C:\Users\debje\Downloads\dataset.json',lines=True) 
df1.head()

df1.shape

df1.dtypes

df1.isnull().mean()*100

df1 = df1.drop(['review_summary','review_text'],axis=1)
df1.tail()

df1.iloc[5:15]

df1.nunique()

df1['new_bust'] = df1['bra size'] - 1
df1.tail()

df1['height'].fillna("0",inplace=True)

df1 = df1.drop(['bust'],axis=1)

#height conversion 

new_height = df1["height"].str.split(" ", n = 1, expand = True) 

df1["feet"]= new_height[0]

df1["inch"]= new_height[1]

df1.drop(columns =["height"],inplace = True)

df1.head()

new_feet = df1["feet"].str.split('ft',n = 1, expand = True)

df1["new_feet"]= new_feet[0]

#df1['new_feet'].fillna("0",inplace=True)

df1.drop(columns =["feet"], inplace = True)


new_feet = df1["inch"].str.split('in',n = 1, expand = True)

df1["new_inch"]= new_feet[0]

df1.drop(columns =["inch"], inplace = True)

#df1['new_inch'].fillna("0",inplace=True)

df1.head()

df1['new_feet'] = df1['new_feet'].astype(float)

df1['new_inch'] = df1['new_inch'].astype(float)

df1.dtypes

inch = df1['new_feet']*12 + df1['new_inch']

df1['height_cms'] = inch * 2.54 

df1 = df1.drop(['new_feet','new_inch'],axis=1)

df1.head()


Visualization 

import seaborn as sns
import matplotlib.pyplot as plt 

sns.set(style='darkgrid')
sns.catplot(aspect=11.7/8.27,x="category",y="hips",col="length",kind="bar",data=df1)

#doubletap on the image to zoom

g = sns.catplot(x="hips", y="category", hue="length",
                height=10.5, aspect=1.5,
                kind="box", legend=False, data=df1);
#g.add_legend(title="Meal")
#g.set_axis_labels("Total bill ($)", "")
#g.set(xlim=(0, 60), yticklabels=["Thursday", "Friday", "Saturday", "Sunday"])
g.despine(trim=True)
g.fig.set_size_inches(6.5, 3.5)
#g.ax.set_xticks([5, 15, 25, 35, 45, 55], minor=True);
plt.setp(g.ax.get_yticklabels(), rotation=30);

#from pandas_profiling import ProfileReport

#profile = ProfileReport(df1,title='pandas profiling report',explorative=True)

df1['waist'].unique()
#there is no outliers so we have to use mean instead of Median to fill na 
    

df1.waist.mean()

df1['waist'].fillna(31, inplace=True)

df1.head()

df1.hips.unique()

df1.hips.mean()

df1['hips'].fillna(40.3,inplace=True) 
df1.head()

df1 = df1.drop(['shoe size','shoe width'],axis=1)
#removing shoe size and shoe width because we are working on women's shirt not women's shoe

df1 = df1.dropna()

#changing small and large fit to unfit because in oth ways they are well fitted to customer so we can take it in one frame

df1['fit'] = df1['fit'].str.replace('small','unfit')
df1['fit'] = df1['fit'].str.replace('large','unfit')
df1.head()

df1.fit.value_counts()

# List of variables to map

varlist =  ['fit']

# Defining the map function
def binary_map(x):
    return x.map({'unfit':1, 'fit':0})

# Applying the function to the housing list
df1[varlist] = df1[varlist].apply(binary_map)

varlist =  ['length']

# Defining the map function
def binary_map(x):
    return x.map({'just right':0, 'slightly long':1, 'very short':2, 'slightly short':3,
       'very long':4, 'fit':5})

# Applying the function to the housing list
df1[varlist] = df1[varlist].apply(binary_map)
df1.head()

sns.heatmap(df1.corr(method='pearson'),annot=True,cmap="GnBu")

Bra_size and bust is highly correlated  with bra size and hips is secondly highest correlated 

# Checking outliers at 25%, 50%, 75%, 90%, 95% and 99%
df1.describe(percentiles=[.25, .5, .75, .90, .95, .99])

From the distribution shown above, you can see that there no outliers in your data. The numbers are gradually increasing.

from sklearn.model_selection import train_test_split

X = df1.drop(['fit','item_id','user_id','category','user_name','cup size'],axis=1)

X.head()

y = df1['fit']

y.head()

X_train,X_test,y_train,t_test = train_test_split(X,y,train_size=0.7,test_size=0.3,random_state=100)

X_train.head()

# Feature Scaling

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train[['waist','size','quality','hips','bra size','length','new_bust','height_cms']] = scaler.fit_transform(X_train[['waist','size','quality','hips','bra size','length','new_bust','height_cms']])

X_train.head()

### Checking the fit Rate
fit = (sum(df1['fit'])/len(df1['fit'].index))*100
fit

we have almost 31% fit rate


import statsmodels.api as sm

# Logistic regression model
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())


log_trained=logm1.fit()

log_trained.summary()

Z - Gives you significance of the variable 

p is hypothesis testing of significance of Z
    H0: varibale is not significant 
    H0: var =0
    H1: var !=0

P high implies that the variable is not significant 

so 
    Lower the p value more the significant of the variable 
    Higher the P value less significant the variable 

# Feature Selection Using RFE

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

from sklearn.feature_selection import RFE
rfe = RFE(logreg, 15)           
rfe = rfe.fit(X_train, y_train)

rfe.support_

list(zip(X_train.columns, rfe.support_, rfe.ranking_))

col = X_train.columns[rfe.support_]

X_train.columns[~rfe.support_]

col

Assessing the model with StatsModels

X_train_sm = sm.add_constant(X_train[col])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()

X_train_sm.columns

# Getting the predicted values on the train set
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]

y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]

y_train_pred_final = pd.DataFrame({'fit':y_train.values, 'fit_Prob':y_train_pred})
y_train_pred_final['UserID'] = y_train.index
y_train_pred_final.head()

y_train_pred_final['predicted'] = y_train_pred_final.fit_Prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head(50)

from sklearn import metrics

# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.fit, y_train_pred_final.predicted )
print(confusion)

# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.fit, y_train_pred_final.predicted))

around 70% accuracy

res.summary()

# from the above solution, we can say that length and size is very important for perfect fitting for the cloth.Alice must focus on both two as the value of z is high in both and both show the high significance

g = sns.catplot(x="length", y="fit", data=df1,
                height=6, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("fitting probability")

g = sns.catplot(x="size", y="fit", data=df1,
                height=6, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("fitting probability")

