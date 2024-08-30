from Importacion import carga

def set_of_columns(data):
    lista_de_elementos = []
    todas_las_columnas = list(data)
    for index in todas_las_columnas:
        lista_de_elementos.append(list(set(data[index])))
    return todas_las_columnas, lista_de_elementos





