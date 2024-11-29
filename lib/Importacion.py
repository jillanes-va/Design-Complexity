from pandas import read_csv, read_excel

def carga(nombre_archivo: str, columnas_importantes: list):
    '''Función que toma el nombre del archivo (que DEBE estar en la carpeta /data) en formato csv e importa las columnas importantes limpiando los NaNs'''
    str_archivo = r'./data/' + nombre_archivo
    data = read_csv(str_archivo).loc[:,columnas_importantes]
    data_sin_nan = data.dropna()
    return data_sin_nan

def locarno(nombre_archivo: str, columnas_importantes: list):
    '''Función que toma el nombre del archivo (que DEBE estar en la carpeta /data) en formato .xlsx'''
    str_archivo = r'./data/TO_DO/locarno2.xlsx'
    hoja = 'categories'
    data = read_excel(str_archivo).loc[:,columnas_importantes]
    return data

def csv_to_list(str_file):
    '''Toma un archivo csv y lo retorna a list con n tuplas dependiendo de la cantidad de columnas'''
    with open(str_file, 'r') as documento:
        datos = [data.split(',') for data in documento.readlines()]
    datos.pop(0)
    lista_pais_ECI = []
    for array in datos:
        lista_pais_ECI.append( (array[1], float(array[2])) )
    return lista_pais_ECI

def csv_to_dict(str_file):
    '''Toma un archivo csv y retorna un dict entre dos columnas'''
    with open(str_file, 'r') as documento:
        datos = [data.split(',') for data in documento.readlines()]
    datos.pop(0)
    lista_pais_ECI = []
    for array in datos:
        array_0 = array[0]
        array_1 = array[1]
        l = len(array_1)
        lista_pais_ECI.append( (array_0, array_1[: l-1 ]) )
    return dict(lista_pais_ECI)