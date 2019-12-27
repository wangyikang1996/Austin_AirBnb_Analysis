#!/usr/bin/env python
# coding: utf-8

# In[43]:


from pandas import Series, DataFrame
import pandas as pd
from patsy import dmatrices
get_ipython().run_line_magic('pylab', 'inline')


# In[45]:


#read in data set
df = pd.read_csv('Dataset/listings.csv')
# pick up all the potential useful col - integer + sequential list + dates and other useful cols
df = df[["host_is_superhost","host_verifications","host_has_profile_pic","host_identity_verified","zipcode",
         "is_location_exact","room_type","accommodates","bathrooms","bedrooms","beds","amenities","price",
         "security_deposit",
         "number_of_reviews","review_scores_rating","instant_bookable","cancellation_policy"]]
print (len(df))
# room_type - dummy cancellation_policy
# NA - square_feet	weekly_price	monthly_price	security_deposit	cleaning_fee review_scores_rating
df.columns.values


# In[46]:


# Explore what's the most important items in amenities
list1 = []
st = ""
for i in df["amenities"]:
    for j in i:
        st += j

special_char = '}{""'
st2 = ''
for char in special_char:
    st2 = st.replace(char,'')

st2 = st2.replace("{","")
l2 = st2.split(",")
dict1 = {}
for i in l2:
    if i not in dict1:
        dict1[i] = 1
    else:
        value = dict1.get(i)
        dict1[i] = value + 1
# sort dirct1
sorted_d = sorted(dict1.items(), key=lambda x: x[1])

# most important key words:
l4 = [('Essentials', 1.0*7509/9663), ('Washer', 1.0*7720/9663), ('Smoke detector', 1.0*7958/9663), ('Free parking on premises', 1.0*8432/9663), ('Wireless Internet', 1.0*8500/9663), ('Kitchen', 1.0*8984/9663), ('Heating', 1.0*9033/9663), ('Air conditioning', 1.0*9224/9663)]

df3 = pd.DataFrame(l4)
df3 = df3.sort_values(by = 1, ascending = False)

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
# Creating a bar plot
sns.barplot(x=df3[1], y=df3[0])
plt.xlim(0., 1.0)
# Add labels to your graph
plt.xlabel('Percentage')
plt.ylabel('Amenities')
plt.title("Amenities Frequency Chart")
plt.legend()
plt.show()

df3


# # Clean data

# In[47]:


# convert t(True) = 1; f(False) = 0;
df["host_is_superhost"] = df["host_is_superhost"].replace('f',0).replace('t',1)
df["host_has_profile_pic"] = df["host_has_profile_pic"].replace('f',0).replace('t',1)
df["host_identity_verified"] = df["host_identity_verified"].replace('f',0).replace('t',1)
df["is_location_exact"] = df["is_location_exact"].replace('f',0).replace('t',1)
df["instant_bookable"] = df["instant_bookable"].replace('f',0).replace('t',1)

# Run the code only once
# Read how many elements inside the host_verifications ex['email', 'phone', 'reviews'] = 3
def count_len(st):
    list1 = st.split(',')
    return len(list1)
df["host_verifications"] = df["host_verifications"].map(count_len)
df["host_verifications"][:5]

# Run the code only once
# apply the same func to amenities {"Cable TV","Wireless Internet","Air conditioning"} - did not really work well if we directly count the 
# number of amenities. Therefore, for the second try, we decided to count several key words:
# TV Internet Air-conditioning Parking 
keylist = ['Essentials','Washer','Smoke detector','Free parking on premises','Wireless Internet','Kitchen','Heating','Air conditioning']
count_list = []
for i in df["amenities"]:
    count = 1
    for j in keylist:
        if j in i:
            count += 1
    count_list.append(count)
count_df =  pd.Series(count_list).astype(int)

df['amenities'] = count_df
df["amenities"][:5]

# Run this code only once
# There are 3 types of room_type: ['Entire home/apt', 'Shared room', 'Private room'], we give them [3,1,2]
def rate_room(room):
    if room == "Entire home/apt":
        return 3
    elif room == "Shared room":
        return 1
    elif room == "Private room":
        return 2

df["room_type"] = df["room_type"].map(rate_room)
df["room_type"] [:10]

# remove $ and other special char, run this code only once
def no_special_char(st):
    special_cha = '$'
    for i in special_cha:
        if i in st:
            return st.replace(i,'0')
def no_comma(st):
        return  st.replace(',', '')
def no_dot(st):
        return  st.replace('.00', '')
def to_int(st):
        return int(st)

#df["cleaning_fee"].fillna(value=pd.np.nan, inplace=True)
# fill na
df["security_deposit"] = df["security_deposit"].fillna('0')
df["bedrooms"] = df["bedrooms"].fillna('0')
df["bathrooms"] = df["bathrooms"].fillna('0')
df["zipcode"] = df["zipcode"].fillna('0')
df.zipcode = df.zipcode.astype(int)
df.bedrooms = df.bedrooms.astype(int)
df.bathrooms = df.bathrooms.astype(int)

df["price"] = df["price"].map(no_special_char)
df["price"] = df["price"].map(no_comma)
df["price"] = df["price"].map(no_dot)
df["price"] = df["price"].map(to_int)

# cancelation col:['strict', 'moderate', 'flexible', 'super_strict_30','super_strict_60'] [3,2,1,4,4]
def cancel_type(type):
    if type == "strict":
        return 3
    elif type == "moderate":
        return 2
    elif type == "flexible":
        return 1
    else:
        return 4
    
df["cancellation_policy"] = df["cancellation_policy"].map(cancel_type)


# # Exploratory Analysis

# In[48]:


df[:10]


# Pricing for all Airbnb listings

# In[49]:


plot(df.index,df['price'],marker ='o', color='blue',linestyle='None')
title('Price Distribution')
xlabel('Listings')
ylabel('Price')


# In[ ]:





# -Lower pricing, More reviews

# In[50]:



df_1 = df.copy()

df_1['price_bin'] = pd.qcut(df_1['price'], [0.0, .71, .85, 1.])
#df_1['price_bin'] = pd.qcut
df_1.groupby(['price_bin'])['number_of_reviews'].agg(['count']).plot.barh()
title('Price Range vs Counts')
ylabel('Price Range')
xlabel('Counts')


# Most of the Super Host has lower price house

# In[51]:


mask = df_1['host_is_superhost'] == 1

df_1[mask].groupby(['price_bin'])['price'].agg(['count']).plot.barh()
title('Price vs SuperHost')
ylabel('Price Range')
xlabel('SuperHost')

df_1[mask].groupby(['price_bin'])['price'].agg(['count'])


# In[52]:


mask = df_1['amenities'] >= 3

df_1['amenities'].value_counts().plot(kind='bar',rot=0)
title('SuperHost vs Amenities')
ylabel('SuperHost')
xlabel('Amenities')

df_1['amenities'].value_counts()


# The relationship between is superhost vs Number of Reviews 

# the higher price,the better review score
# - in high price houses, guests and hosts tend to respect to each other
# - Therefore, the high price hosts usually have higher review score.

# In[53]:


df_1['review_score_bin'] = pd.cut(df_1['review_scores_rating'], [0,99,100])
df_price_review = df_1.groupby(['price_bin','review_score_bin'])['review_scores_rating'].agg(['count'])

percent_review = np.array(['0.568385','0.431615','0.453911','0.546089','0.405253','0.594747'])
percentage = pd.Series(percent_review, index =df_price_review.index) 
#print percentage
low_review = [0.568385, 0.453911, 0.405253]
high_review = [0.431615, 0.546089, 0.594747]
index = ['Low Price Listings', 'Medium Price Listings', 'High Price Listings']
df_123 = pd.DataFrame({'High Review Score': high_review,'Low Review Score': low_review}, index=index)
ax = df_123.plot.bar(rot=0)

title('Price vs Review Score')
ylabel('%')
xlabel('Price Category')


# Bedroom distribution, private bedroom leads to better review ratings, which ultimately leads to higher price?

# In[54]:


df_1['bedrooms'].value_counts().sort_index().plot(kind="bar")
ylabel('Count')
xlabel('Number of bedroom')


# In[55]:


bed_review = df_1.groupby(['bedrooms'])['review_scores_rating'].agg(['mean']).plot(kind = 'bar',rot=0).set_ylim(95,100)
xlabel('Number of bedroom')
ylabel('Review Score')
bath_review = df_1.groupby(['bathrooms'])['review_scores_rating'].agg(['mean']).plot(kind = 'bar',rot=0).set_ylim(95,100)
xlabel('Number of bathroom')
ylabel('Review Score')


# Cancelation Policy andd price    

# # Logistic Regression

# In[56]:


# we split the price 
df.price.quantile(0.75)

formula = 'target ~ 0 +host_is_superhost + host_verifications + host_has_profile_pic + host_identity_verified ' +            '+ is_location_exact + accommodates + bathrooms + bedrooms + beds' +           '+amenities'+         '+number_of_reviews + review_scores_rating + instant_bookable + C(cancellation_policy)'      
        
# price not included C(room_type)
df['target'] = 0.0
df['target'][df['price'] >= 300] = 1.0 # if >= 300, high price
df['target'].value_counts()

Y, X = dmatrices(formula, df, return_type='dataframe')
y = Y['target'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Fit the classifier
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
result = model.fit(X_train, y_train)


# In[57]:


from sklearn import metrics
#train accuracy
prediction_train = model.predict(X_train)
print (metrics.accuracy_score(y_train, prediction_train))


# In[58]:


#predict accuracy
prediction = model.predict(X_test)
print (metrics.accuracy_score(y_test, prediction))


# In[59]:


# y_train is 0 or 1.
print ('Number of positive examples =', len(y_train[y_train==1]))
print ('Number of negative examples =', len(y_train[y_train==0]))

negative_examples_in_test = len(y_test[y_test==0])
total_examples_in_test = len(y_test)

print ('Number of examples where baseline is correct =', negative_examples_in_test)
print ('Baseline accuracy =', negative_examples_in_test * 1.0 / total_examples_in_test)


# In[60]:


model.coef_


# In[61]:


model.intercept_


# In[62]:


weights = Series(model.coef_[0],
                 index=X.columns.values)


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
# Creating a bar plot
sns.barplot(x=weights.sort_values(ascending = True), y=weights.sort_values().index)
# Add labels to your graph
plt.xlabel('Feature Coef')
plt.ylabel('Features')
plt.title("Visualizing Coef")
plt.legend()
plt.show()

weights.sort_values()


# **Observations**
# 
# *  **room_type** 
#     * 1 = shared
#     * 2 = private
#     * 3 = entire
# * **cancellation_policy** 
#     * ['strict', 'moderate', 'flexible', 'super_strict_30','super_strict_60'] [3,2,1,4,4] 
# 

# # Tree Method

# In[63]:


from pandas import Series, DataFrame
import pandas as pd
from patsy import dmatrices
import warnings
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

get_ipython().run_line_magic('pylab', 'inline')
warnings.filterwarnings('ignore')


# In[64]:


print (len(df))
df.columns.values


# In[65]:



df_4 = df.copy()

df_4['price_bin'] = pd.qcut(df_4['price'], [0, 0.20, 0.40, 0.60, 0.80,1.0])
df_4.groupby(['price_bin'])['number_of_reviews'].agg(['count']).plot.barh()
title('Price Range vs Counts')
ylabel('Price Range')
xlabel('Counts')


# In[66]:



def cate_price(price):
    if price <= 200:
        return 1
    elif price <= 400:
        return 2
    else:
        return 3
    
df['price'] = df['price'].map(cate_price)


# In[67]:


df['price'] [:5]


# In[ ]:





# In[68]:



formula = 'price ~ 0 +host_is_superhost + host_verifications + host_has_profile_pic + host_identity_verified ' +            '+ is_location_exact + accommodates + bathrooms + bedrooms + beds' +           '+amenities'+         '+number_of_reviews + review_scores_rating + instant_bookable + C(cancellation_policy)'#print formula
Y, X = dmatrices(formula, df, return_type='dataframe')
y = Y['price'].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# # Decision Tree Depth

# In[69]:


scores_train = {}
scores_test = {}
for depth in range(1, 11):
    model =  tree.DecisionTreeClassifier(criterion='entropy' ,max_depth = depth)
    model.fit(X_train, y_train)
    
    prediction_test = model.predict(X_test)
    accuracy_test = metrics.accuracy_score(y_test, prediction_test)
    scores_test[depth] = accuracy_test
    
    prediction_train = model.predict(X_train)
    accuracy_train = metrics.accuracy_score(y_train, prediction_train)
    scores_train[depth] = accuracy_train
    
Series(scores_train).sort_index().plot()
xlabel('Depth of trees in ensemble')
ylabel('Test accuracy')

Series(scores_test).sort_index().plot()
xlabel('Depth of trees in ensemble')
ylabel('Test accuracy')

title('Decision Tree Depth')


# # Bagging Depth

# In[70]:


scores_train = {}
scores_test = {}
for depth in range(1, 11):
    model =  BaggingClassifier(tree.DecisionTreeClassifier(criterion='entropy',max_depth = depth), random_state=1)
    model.fit(X_train, y_train)
    
    prediction_test = model.predict(X_test)
    accuracy_test = metrics.accuracy_score(y_test, prediction_test)
    scores_test[depth] = accuracy_test
    
    prediction_train = model.predict(X_train)
    accuracy_train = metrics.accuracy_score(y_train, prediction_train)
    scores_train[depth] = accuracy_train
    
Series(scores_train).sort_index().plot()
xlabel('Depth of trees in ensemble')
ylabel('Test accuracy')

Series(scores_test).sort_index().plot()
xlabel('Depth of trees in ensemble')
ylabel('Test accuracy')

title('Bagging Depth')


# # RF

# In[71]:


scores_train = {}
scores_test = {}
for depth in range(1, 11):
    model =  RandomForestClassifier(n_estimators=200,max_depth = depth, random_state=1)
    model.fit(X_train, y_train)
    
    prediction_test = model.predict(X_test)
    accuracy_test = metrics.accuracy_score(y_test, prediction_test)
    scores_test[depth] = accuracy_test
    
    prediction_train = model.predict(X_train)
    accuracy_train = metrics.accuracy_score(y_train, prediction_train)
    scores_train[depth] = accuracy_train
    
Series(scores_train).sort_index().plot()
xlabel('Depth of trees in ensemble')
ylabel('Test accuracy')

Series(scores_test).sort_index().plot()
xlabel('Depth of trees in ensemble')
ylabel('Test accuracy')

title('Random Forest Depth')


# In[72]:


scores = {}
for num_trees in [50, 100, 200, 300, 400]:
    model = RandomForestClassifier(n_estimators=num_trees,max_depth = depth, random_state=1)
    model.fit(X_train, y_train)
    prediction_test = model.predict(X_test)
    accuracy_test = metrics.accuracy_score(y_test, prediction_test)
    scores[num_trees] = accuracy_test
Series(scores).sort_index().plot()
xlabel('Number of trees in ensemble')
ylabel('Test accuracy')

title('Random Forest Num Tree')


# # G

# In[73]:


scores_train = {}
scores_test = {}
for depth in range(1, 5):
    model =  GradientBoostingClassifier(n_estimators=200, max_depth=depth, random_state=1)
    model.fit(X_train, y_train)
    
    prediction_test = model.predict(X_test)
    accuracy_test = metrics.accuracy_score(y_test, prediction_test)
    scores_test[depth] = accuracy_test
    
    prediction_train = model.predict(X_train)
    accuracy_train = metrics.accuracy_score(y_train, prediction_train)
    scores_train[depth] = accuracy_train
    
Series(scores_train).sort_index().plot()
xlabel('Depth of trees in ensemble')
ylabel('Test accuracy')

Series(scores_test).sort_index().plot()
xlabel('Depth of trees in ensemble')
ylabel('Test accuracy')

title('Boosting Depth')


# In[74]:


scores = {}
for num_trees in [50, 100, 200, 300, 400]:
    model = GradientBoostingClassifier(n_estimators=num_trees, max_depth=2, random_state=1)
    model.fit(X_train, y_train)
    prediction_test = model.predict(X_test)
    accuracy_test = metrics.accuracy_score(y_test, prediction_test)
    scores[num_trees] = accuracy_test
Series(scores).sort_index().plot()
xlabel('Number of trees in ensemble')
ylabel('Test accuracy')


title('Boosting Num Tree')


# In[75]:


#All the models we want to test out, in one list.
model_decision = tree.DecisionTreeClassifier(criterion='entropy' ,max_depth = 5)
model_bagging = BaggingClassifier(tree.DecisionTreeClassifier(criterion='entropy',max_depth = 2), random_state=1)
model_rf = RandomForestClassifier(n_estimators=210, max_depth = 5, random_state=1)
model_g =  GradientBoostingClassifier(n_estimators=120, max_depth=1, random_state=1)
#for (name, model) in model_list:
#    print 'Fitting', name
model_decision.fit(X_train, y_train)
model_bagging.fit(X_train, y_train)
model_rf.fit(X_train, y_train)
model_g.fit(X_train, y_train)
from sklearn import metrics


# In[76]:


# model_decision
prediction_train = model_decision.predict(X_train)
accuracy_train_1 = metrics.accuracy_score(y_train, prediction_train)
prediction_test = model_decision.predict(X_test)
accuracy_test_1 = metrics.accuracy_score(y_test, prediction_test)
print ('Accuracy for Decision Tree')
print ('Train:',accuracy_train_1)
print ('Test:',accuracy_test_1)
print()
# model_bagging
prediction_train = model_bagging.predict(X_train)
accuracy_train_2 = metrics.accuracy_score(y_train, prediction_train)
prediction_test = model_bagging.predict(X_test)
accuracy_test_2 = metrics.accuracy_score(y_test, prediction_test)
print ('Accuracy for Bagging')

print ('Train:',accuracy_train_2)
print ('tTest:',accuracy_test_2)
print()

# model_rf
prediction_train = model_rf.predict(X_train)
accuracy_train_3 = metrics.accuracy_score(y_train, prediction_train)
prediction_test = model_rf.predict(X_test)
accuracy_test_3 = metrics.accuracy_score(y_test, prediction_test)
print ('Accuracy for RF')

print ('Train:',accuracy_train_3)
print ('Test:',accuracy_test_3)
print
# model_g
prediction_train = model_g.predict(X_train)
accuracy_train_4 = metrics.accuracy_score(y_train, prediction_train)
prediction_test = model_g.predict(X_test)
accuracy_test_4 = metrics.accuracy_score(y_test, prediction_test)
print ('Accuracy for Boosting')

print ('Train:',accuracy_train_4)
print ('Test:',accuracy_test_4)

Baseline_accuracy = 0.60
Baseline_accuracy2 = 0.60

train= [accuracy_train_1, accuracy_train_2,accuracy_train_4,accuracy_train_3,Baseline_accuracy]
test = [accuracy_test_1, accuracy_test_2, accuracy_test_4,accuracy_test_3,Baseline_accuracy2]
index = ['Decision Tree', 'Bagging', 'Boosting','Random Forest','Baseline']
df_123 = pd.DataFrame({'train accuracy': train,'test accuracy': test}, index=index)
ax = df_123.plot.bar(rot=0).set_ylim(0.0,0.80)

title('Tree Models vs Accuracy')
ylabel('')
xlabel('')


# In[77]:


accuracy_train_1


# In[78]:


import pandas as pd

# Create Series for coef
thelist = [ ['host_is_superhost'], ['host_verifications'], ['host_has_profile_pic'],["host_identity_verified"],["is_location_exact"],["accommodates"],["bathrooms"],["bedrooms"],["beds"],["amenities"],["number_of_reviews"],["review_scores_rating"],["instant_bookable"],["cancellation_policyL1"],["cancellation_policyL2"],["cancellation_policyL3"],["cancellation_policyL4"]]
df_2 = pd.Series( (v[0] for v in thelist) )


# In[79]:


# Decision Tree Coef
feature_imp = pd.Series(model_decision.feature_importances_,index=df_2).sort_values(ascending=False)
feature_imp


# In[81]:


# tf coef
feature_imp = pd.Series(model_rf.feature_importances_,index=df_2).sort_values(ascending=False)
print (feature_imp)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()


# In[82]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('pylab', 'inline')
list3 = [['Train Accuracy',0.872],['Test Accuracy',0.878],['Baseline Accuracy',0.794]]
df5 = pd.DataFrame(list3)
df5 = df5.rename(columns={0: "Dataset", 1: "Accuracy"})


# In[83]:


df5.set_index('Dataset').plot(kind='bar',rot=0)
ylabel('Accuracy')
title('Accuracy Chart')


# In[ ]:




