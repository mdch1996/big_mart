import pandas as pd
import category_encoders as ce
from fancyimpute import KNN
from sklearn import datasets, linear_model

from pre_processing import pre_processor


pd.set_option('display.max_columns', 50)

df_train = pd.read_csv("data/Train_UWu5bXk.csv")
train_target_value = df_train['Item_Outlet_Sales']

train_training_data = df_train.drop('Item_Outlet_Sales', 1)
train_training_data = pre_processor(train_training_data)

lm = linear_model.LinearRegression()
model = lm.fit(train_training_data, train_target_value)

df_test = pd.read_csv("data/Test_u94Q5KV.csv")
test_training_data = pre_processor(df_test)

predictions = lm.predict(test_training_data)
predictions = pd.DataFrame(data=predictions, columns=['Item_Outlet_Sales'])

data = {
    'Item_Identifier': df_test['Item_Identifier'],
    'Outlet_Identifier': df_test['Outlet_Identifier'],
    'Item_Outlet_Sales': predictions['Item_Outlet_Sales']
}
df_final = pd.DataFrame(data=data)
df_final.to_csv('data/test_final.csv', index=False)
