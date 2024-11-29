from pandas import read_csv, read_excel
from pyreadstat import read_dta

def carga(nombre_archivo: str, columnas_importantes: list):
    '''Función que toma el nombre del archivo (que DEBE estar en la carpeta /data) en formato csv e importa las columnas importantes limpiando los NaNs'''
    str_archivo = r'./data/' + nombre_archivo
    data = read_csv(str_archivo).loc[:,columnas_importantes]
    data_sin_nan = data.dropna()
    return data_sin_nan

def carga_especial(nombre_archivo: str, columnas_importantes: list):
    '''Función que toma el nombre del archivo (que DEBE estar en la carpeta /data) en formato csv e importa las columnas importantes limpiando los NaNs'''
    str_archivo = r'./data/' + nombre_archivo
    data, meta = read_dta(str_archivo)
    data = data.loc[:, columnas_importantes ]
    data_sin_nan = data.loc[data['exporter'] != 'World'].dropna()
    return data_sin_nan


def locarno(nombre_archivo: str, columnas_importantes: list):
    '''Función que toma el nombre del archivo (que DEBE estar en la carpeta /data) en formato .xlsx'''
    str_archivo = r'./data/TO_DO/locarno2.xlsx'
    hoja = 'categories'
    data = read_excel(str_archivo).loc[:,columnas_importantes]
    return data