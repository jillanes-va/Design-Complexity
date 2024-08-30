import numpy as np

def ordenacion(data):
    datos_nombre_diseñador = data['designer_name'].values  # En formato np.array
    datos_pais_diseñador = data['designer_country'].values
    datos_categoria_premio = data['award_category'].values
    datos_periodo_premio = data['award_period'].values
    datos_puntaje_premio = data['award_score'].values

    lista_nombres = []
    lista_categorias = []
    lista_paises = []
    lista_periodos = []
    lista_puntajes = []