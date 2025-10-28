import pandas as pd
df = pd.read_excel('data_enhance.xlsx', engine='openpyxl')
print(f'Rows: {len(df)}, Columns: {len(df.columns)}')
print(f'Columns: {list(df.columns)}')
df.to_csv('data_enhance.csv', index=False, encoding='utf-8-sig')
print('Done! Saved to data_enhance.csv')

