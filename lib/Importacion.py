from pandas import read_csv

def carga(nombre_archivo: str, columnas_importantes: list):
    '''Función que toma el nombre del archivo (que DEBE estar en la carpeta /Datos) en formato csv y e immporta las columnas importantes limpiando los NaNs'''
    str_archivo = r'./data/' + nombre_archivo
    data = read_csv(str_archivo).loc[:,columnas_importantes]
    data_sin_nan = data.dropna()
    return data_sin_nan
