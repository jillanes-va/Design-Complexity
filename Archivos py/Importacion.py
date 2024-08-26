import pandas as pd

#%% Cargamos los datos y limpiamos los NaNs

str_archivo = r'../Datos/wrd_04_all-data.csv'

data = pd.read_csv(str_archivo)
data_sin_nan = data.loc[data['award_category'].notna() & data['designer_country'].notna() & data['award_period'].notna()]

print(data_sin_nan.columns) #Printea todas las columnas

