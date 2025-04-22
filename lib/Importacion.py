from pandas import read_csv, read_excel
from pyreadstat import read_dta
from numpy import nan

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

def carga_excel(nombre_archivo:str, columnas_importantes: list):
    '''Función que importa directamente datasets en excel (.xls)'''
    str_archivo = r'./data/datasets/' + nombre_archivo
    data = read_excel(str_archivo).replace('no data', nan).loc[:, columnas_importantes]
    return data

def locarno(nombre_archivo: str, columnas_importantes: list):
    '''Función que toma el nombre del archivo (que DEBE estar en la carpeta /data) en formato .xlsx'''
    str_archivo = r'./data/TO_DO/locarno2.xlsx'
    hoja = 'categories'
    data = read_excel(str_archivo).loc[:,columnas_importantes]
    return data

def dictionary_from_csv(nombre_archivo: str, ranking = False):
    '''Función que lee un archvio csv con dos columnas y lo transforma en un diccionario'''
    location_str = r'./data/' + nombre_archivo
    dictionary = {}
    with open(location_str, 'r', encoding = 'utf-8') as awards_gdp:
        awards_gdp.readline()
        if ranking:
            for line in awards_gdp:
                rank, key, value = line.strip().split(',')
                dictionary.update({key: float(value)})
        else:
            for line in awards_gdp:
                key, value = line.strip().split(',')
                dictionary.update({key:value})
    return dictionary