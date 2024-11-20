from pandas import read_csv, read_excel

def carga(nombre_archivo: str, columnas_importantes: list):
    '''Función que toma el nombre del archivo (que DEBE estar en la carpeta /data) en formato csv e importa las columnas importantes limpiando los NaNs'''
    str_archivo = r'./data/' + nombre_archivo
    data = read_csv(str_archivo).loc[:,columnas_importantes]
    data_sin_nan = data.dropna()
    return data_sin_nan

def locarno(nombre_archivo: str, columnas_importantes: list):
    '''Función que toma el nombre del archivo (que DEBE estar en la carpeta /data) en formato .xlsx'''
    str_archivo = r'./data/Por ordenar/locarno2.xlsx'
    hoja = 'categories'
    data = read_excel(str_archivo).loc[:,columnas_importantes]
    return data
