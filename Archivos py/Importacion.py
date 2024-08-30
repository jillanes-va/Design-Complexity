from pandas import read_csv


#str_archivo = r'../Datos/wrd_04_all-data.csv'
#%% Cargamos los datos y limpiamos los NaNs

def carga(nombre_archivo: str):
    '''Función que toma el nombre del archivo (que DEBE estar en la carpeta /Datos)'''
    str_archivo = r'../Datos/' + nombre_archivo
    data = read_csv(str_archivo)
    data_sin_nan = data.loc[
        data['award_category'].notna() & data['designer_country'].notna() & data['award_period'].notna()]
    return data_sin_nan
