import pandas as pd
import category_encoders as ce
from fancyimpute import KNN

pd.set_option('display.max_columns', 50)
df = pd.read_csv("data/Train_UWu5bXk.csv")

# ############################ Encoding Section ########################

# -----------common scripts---------
# print(df.head())
# print(df.isnull().sum())
# df.info()
# print(df['Item_Fat_Content'].unique())
# print(df['Item_Identifier'].value_counts())


# -----------'Item_Identifier'----------------
encoder = ce.BinaryEncoder(cols=['Item_Identifier'])
df = encoder.fit_transform(df)


# -----------'Item_Weight'----------------
# df = KNN(k=5).fit_transform(df)


# -----------'Item_Fat_Content'---------------
df['Item_Fat_Content'] = df['Item_Fat_Content'].astype('category').replace({'Low Fat': 0,
                                                                            'low fat': 0,
                                                                            'LF': 0,
                                                                            'Regular': 1,
                                                                            'reg': 1})


# -----------'Item_Visibility'---------------
# print(df['Item_Visibility'].value_counts())


# -----------'Item_Type'---------------
encoder = ce.BinaryEncoder(cols=['Item_Type'])
df = encoder.fit_transform(df)


# -----------'Item_MRP'---------------
# print(df['Item_MRP'].value_counts())


# -----------'Outlet_Identifier'---------------
encoder = ce.BinaryEncoder(cols=['Outlet_Identifier'])
df = encoder.fit_transform(df)


# -----------'Outlet_Establishment_Year'---------------
# print(df['Outlet_Establishment_Year'].value_counts())


# -----------'Outlet_Size'---------------
df['Outlet_Size'] = df['Outlet_Size'].replace({'Small': 0, 'Medium': 1, 'High': 3})



# -----------'Outlet_Location_Type'---------------
print(df['Outlet_Location_Type'].value_counts())
# df['Outlet_Location_Type'] = df['Outlet_Location_Type'].replace({'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2})


# -----------'Outlet_Type'---------------
print(df['Outlet_Type'].value_counts())
encoder = ce.BinaryEncoder(cols=['Outlet_Type'])
df = encoder.fit_transform(df)


# -----------'Item_Outlet_Sales'---------------
# print(df['Item_Outlet_Sales'].value_counts())




# ####################### Impute Section #######################

# -----------'Item_Weight'----------------
# data = KNN(k=5).fit_transform(df)
# print(data.shape)
# print("-------------")
# print(len(df.iloc[0:0, :]))
# df_modified = pd.DataFrame(data=data)
# df.iloc[1:, :] = df_modified
# df.to_csv('data/train_modified.csv')
# print(df.shape)
# df_modified.info()
# print(df_modified.head())
