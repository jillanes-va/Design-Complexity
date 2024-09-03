from pandas import read_csv

#nombre_archivo = r'wrd_04_all-data.csv'
#columnas = ['designer_country', 'award_category', 'award_period', 'award_score']

def carga(nombre_archivo: str, columnas_importantes: list):
    '''Función que toma el nombre del archivo (que DEBE estar en la carpeta /Datos) en formato csv y e immporta las columnas importantes limpiando los NaNs'''
    str_archivo = r'../Datos/' + nombre_archivo
    data = read_csv(str_archivo).loc[:,columnas_importantes].dropna()
    return data
