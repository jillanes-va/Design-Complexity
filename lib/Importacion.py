from pandas import read_csv, read_excel
from pyreadstat import read_dta
from numpy import nan, isnan
from lib.Tratamiento import inv_dict
import os

def carga(nombre_archivo, columnas_importantes = None, encoding = 'uft-8'):
    '''Función que toma el nombre del archivo (que DEBE estar en la carpeta /data) en formato csv e importa las columnas importantes limpiando los NaNs'''
    str_archivo = r'./data/' + nombre_archivo
    data = read_csv(str_archivo).reset_index()
    if columnas_importantes is not None:
        data = data.loc[:,columnas_importantes]
    data_sin_nan = data.dropna()
    return data_sin_nan

def carga_especial(nombre_archivo, columnas_importantes = None, encoding = 'utf-8'):
    '''Función que toma el nombre del archivo (que DEBE estar en la carpeta /data) en formato csv e importa las columnas importantes limpiando los NaNs'''
    str_archivo = r'./data/' + nombre_archivo
    data = read_csv(str_archivo).reset_index()
    if columnas_importantes is not None:
        data = data.loc[:,columnas_importantes]
    data_sin_nan = data.groupby(columnas_importantes[:-1]).sum().reset_index()
    return data_sin_nan

def carga_excel(nombre_archivo:str, columnas_importantes = None, last = False):
    '''Función que importa directamente datasets en excel (.xls)'''
    str_archivo = r'./data/datasets/' + nombre_archivo
    if columnas_importantes is None:
        data = read_excel(str_archivo).replace('n/a', nan)
    else:
        data = read_excel(str_archivo).replace('n/a', nan).loc[:, columnas_importantes]

    if last:
        first_column = data.columns[0]
        last_column = data.columns[-1]
        return data.loc[:, [first_column, last_column]]
    else:
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

def guardado_ranking(X, dicc, folder, subfolders, name, metric_name):
    num_carac = inv_dict(dicc)
    N = len(X[:, 0])
    M = len(X[0, :])
    for i in range(M):
        X_i = X[:, i]
        conjunto_raro = []
        conjunto_bueno = []
        for j in range(N):
            tupla = (X_i[j], num_carac[j])
            if isnan(tupla[0]):
                conjunto_raro.append(tupla)
            else:
                conjunto_bueno.append(tupla)
        paises_ECI_ = sorted(conjunto_bueno, key=lambda A: A[0], reverse=True)
        sorteado = paises_ECI_ + conjunto_raro
        str_file = r'./data/results/' + folder + r'/'
        if (i != M-1):
            str_file +=  subfolders[i]
        if not os.path.exists(str_file):
            os.mkdir(str_file)

        with open(str_file + '/'+ name +'.csv', 'w+', encoding='utf-8') as f:
            f.writelines(f'rank\t{metric_name}\tcategory\n')
            for k in range(N):
                f.writelines(f'{k + 1}\t{sorteado[k][0]}\t{sorteado[k][1]}\n')
