# hotel-booking-2
#prediction of hotel data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.listdir()
df=pd.read_csv('C:\\Users\\Soura\\Downloads\\hotel_bookings.csv')
df.head()
df.shape
df.isna().sum()
def data_clean(df):
    df.fillna(0,inplace=True)
    print(df.isnull().sum())
data_clean(df)
df.columns
list=['adults', 'children', 'babies']
for i in list:
    print('{} has unique values as{}'.format(i,df[i].unique()))
filter=(df['children']==0)&(df['adults']==0)&(df['babies']==0)
df[filter]
pd.set_option('display.max_columns',32)
filter=(df['children']==0)&(df['adults']==0)&(df['babies']==0)
df[filter]
data=df[~filter]
data.head()
country_wise_data=data[data['is_canceled']==0]['country'].value_counts().reset_index()
country_wise_data.columns=['country','No of guests']
country_wise_data
!pip install folium
import folium
from folium.plugins import HeatMap
basemap=folium.Map()
!pip install plotly
import plotly.express as px
map_guest= px.choropleth(country_wise_data,
             locations=country_wise_data['country'],
             color=country_wise_data['No of guests'],
             hover_name=country_wise_data['country'],
             title='Home country of guests'
             )
map_guest.show()         

data.head()
data2=data[data['is_canceled']==0]
data2.columns
plt.figure(figsize=(12,8))
sns.boxplot(x='reserved_room_type',y='adr',
           hue='hotel',data=data2)
plt.title('price of room types per night & per person')
plt.xlabel('Room Type')
plt.ylabel('price(euro)')
plt.legend()
plt.show()
data_resort=data[(data['hotel']=='Resort Hotel')&(data['is_canceled']==0)]
data_city=data[(data['hotel']=='City Hotel')&(data['is_canceled']==0)]
data_resort.head()
resort_hotel=data_resort.groupby(['arrival_date_month'])['adr'].mean().reset_index()
resort_hotel
city_hotel=data_city.groupby(['arrival_date_month'])['adr'].mean().reset_index()
city_hotel

final=resort_hotel.merge(city_hotel,on='arrival_date_month')
final.columns=['month','price_for_resort','price_for_city_hotel']
final
!pip install sort-dataframeby-monthorweek
!pip install sorted-months-weekdays
import sort_dataframeby_monthorweek as sd
def sort_data(df,colname):
    return sd.Sort_Dataframeby_Month(df,colname)

final=sort_data(final,'month')
final
final.columns
px.line(final,x='month',y=['price_for_resort','price_for_city_hotel'],title='Room price per night over the months')
data_resort.head()
rush_resort=data_resort['arrival_date_month'].value_counts().reset_index()
rush_resort.columns=['month','no of guests']
rush_resort
rush_city=data_city['arrival_date_month'].value_counts().reset_index()
rush_city.columns=['month','no of guests']
rush_city
final_rush=rush_resort.merge(rush_city,on='month')
final_rush
final_rush.columns=['month','no of guests in resort','no of guests in city hotel']
final_rush
final_rush=sort_data(final_rush,'month')
final_rush
final_rush.columns
px.line(final_rush,x='month',y=['no of guests in resort', 'no of guests in city hotel'],title='Total no of guests per month')
data.head()
data.corr()
co_relation=data.corr()['is_canceled']
co_relation
co_relation.abs().sort_values(ascending=False)
data.groupby('is_canceled')['reservation_status'].value_counts()
list_not=['days_in_waiting_list','arrival_date_year']
num_features=[col for col in data.columns if data[col].dtype!='O'and col not in list_not]
num_features
data.columns
cat_not=['arrival_date_year','assigned_room_type','booking_changes','reservation_status','country','days_in_waiting_list']
cat_not
cat_features=[col for col in data.columns if data[col].dtype=='O'and col not in cat_not]
cat_features
data_cat=data[cat_features]
data_cat.head()
data_cat.dtypes
import warnings
from warnings import filterwarnings
filterwarnings('ignore')

data_cat['reservation_status_date']=pd.to_datetime(data_cat['reservation_status_date'])
data_cat['year']=data_cat['reservation_status_date'].dt.year
data_cat['month']=data_cat['reservation_status_date'].dt.month
data_cat['day']=data_cat['reservation_status_date'].dt.day
data_cat.head()
data.dtypes
data_cat.drop('reservation_status_date',axis=1,inplace=True)
data_cat['cancellation']=data['is_canceled']
data_cat.head()
data_cat['market_segment'].unique()
cols=data_cat.columns[0:8]
cols
data_cat.groupby(['hotel'])['cancellation'].mean()
for col in cols:
    print(data_cat.groupby([col])['cancellation'].mean())
    print('\n')
for col in cols:
    dict=data_cat.groupby([col])['cancellation'].mean().to_dict()
    data_cat[col]=data_cat[col].map(dict)
data_cat.head()
dataframe=pd.concat([data_cat,data[num_features]],axis=1)
dataframe.head()
dataframe.drop('cancellation',axis=1,inplace=True)
dataframe.shape
dataframe.head()
sns.distplot(dataframe['lead_time'])
import numpy as np
def handle_outlier(col):
    dataframe[col]=np.log1p(dataframe[col])
handle_outlier('lead_time')
sns.distplot(dataframe['lead_time'])
sns.distplot(dataframe['adr'])
handle_outlier('adr')
sns.distplot(dataframe['adr'].dropna())
dataframe.isnull().sum()
dataframe.dropna(inplace=True)
dataframe.drop('is_canceled',axis=1)
y=dataframe['is_canceled']
x=dataframe.drop('is_canceled',axis=1)
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
feature_sel_model=SelectFromModel(Lasso(alpha=0.005,random_state=0))
feature_sel_model.fit(x,y)
feature_sel_model.get_support()
cols=x.columns
selected_feat=cols[feature_sel_model.get_support()]
print('total_features{}'.format(x.shape[1]))
print('selected_features{}'.format(len(selected_feat)))
selected_feat
x=x[selected_feat]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)
y_pred
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,y_pred)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
from sklearn.model_selection import cross_val_score
score=cross_val_score(logreg,x,y,cv=10)
score.mean()
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

models=[]

models.append(('LogisticRegression',LogisticRegression()))
models.append(('Naive bayes', GaussianNB))
models.append(('RandomForest',RandomForestClassifier))
models.append(('Decision Tree', DecisionTreeClassifier()))
models.append(('KNN', KNeighborsClassifier()))

for name,model in  models:
    print(name)
    model.fit(x_train,y_train)
    
    predictions=model.predict(x_test)
    
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(predictions,y_test))
    print('\n')
    
    print(accuracy_score(predictions,y_test))
    print('\n')
