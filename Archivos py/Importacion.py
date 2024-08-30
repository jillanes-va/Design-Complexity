from pandas import read_csv
#nombre_archivo = r'wrd_04_all-data.csv'

def carga(nombre_archivo: str, columnas_importantes: list):
    '''Función que toma el nombre del archivo (que DEBE estar en la carpeta /Datos) en formato csv y lo importa limpiando los NaNs de las columnas importantes'''
    str_archivo = r'../Datos/' + nombre_archivo
    data = read_csv(str_archivo).dropna(subset = columnas_importantes)
    return data