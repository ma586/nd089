import pandas as pd

google_stock = pd.read_csv('./financial_datasets/GOOG.csv')
print(len(google_stock))
# print(google_stock)
print(google_stock.head(10))

print('Google_stock is of type:', type(google_stock))
print('Google_stock has shape:', google_stock.shape)
print(google_stock.isnull().any())
print(google_stock['Low'].describe())
# print(google_stock.corr())


company = pd.read_csv('financial_datasets/fake_company.csv')
company.head()
print(company.groupby(['Year'])['Salary'].sum())