import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
import pickle

################################LIMPIEZA DE DATOS############################################

def conteo_categorias(dataframe):
    """Esta función tiene como objetivo entregar datos acerca de la cantidad de valores o clases diferentes de una variable.

    Parameters
    ----------
    dataframe : [pd.DataFrame]
        DataFrame con las variables a analizar.
           

    Returns
    -------
    [pd.DataFrame]
        Retorna un DataFrame con datos sobre la cantidad de variables.
        
    """

    datos = {
        'variable' : dataframe.columns,
        'valores/categorías' : [len(dataframe[colname].value_counts()) for colname, serie in dataframe.iteritems()],
    }
    
    df_conteo_categorias = pd.DataFrame(datos)
    df_conteo_categorias = df_conteo_categorias.sort_values(ascending=True, by='valores/categorías')
    
    return df_conteo_categorias


def datos_nulos_analisis(dataframe):
    """Esta función tiene como objetivo entregar datos acerca de la cantidad y proporción de los datos nulos de las distintas variables del DataFrame.

    Parameters
    ----------
    dataframe : [pd.DataFrame]
        DataFrame con los datos nulos a analizar.
           

    Returns
    -------
    [pd.DataFrame]
        Retorna un DataFrame con datos sobre los valores perdidos.
        
    """
    
    datos = {
        'variable' : dataframe.columns,
        'cantidades' : [dataframe[colname].value_counts(dropna = False).get(np.nan) for colname, serie in dataframe.iteritems()],
        'porcentajes' : [dataframe[colname].value_counts('%', dropna = False).get(np.nan) for colname, serie in dataframe.iteritems()]
    }
    
    df_analisis_nulos = pd.DataFrame(datos).replace(np.nan, 0)
    df_analisis_nulos = df_analisis_nulos.sort_values(ascending=False, by='cantidades')
    
    return df_analisis_nulos

######################################ANÁLISIS EXPLORATORIO#####################################################

def hist_and_box(dataframe, columna):
    """Esta función tiene como objetivo realizar un boxplot y un histograma con la distribucion de una variable continua.

    Parameters
    ----------
    dataframe : [pd.DataFrame]
        DataFrame con la variable a graficar.
    
    columna : [str]
        Nombre de la variable a graficar
           

    Returns
    -------
    None
        
    """
    f, ax = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.20, .80)})

    sns.boxplot(dataframe[columna], ax=ax[0]).set(title = f'Boxplot de la variable {columna}', xlabel='')
    sns.distplot(dataframe[columna], ax=ax[1])
    ax[1].axvline(np.mean(dataframe[columna]), color='red')
    #plt.legend()
    plt.xlabel(f'Valores de la variable {columna}')
    plt.ylabel('Frecuencia observada')
    plt.title(f'Distribución de la variable {columna}')
    plt.legend(labels=['Curva de densidad',f'Media: {round(dataframe[columna].mean(),2)}', 'Distribucion de los datos'])
    plt.tight_layout()
    
    plt.show();


def barhplot(dataframe, columna, n_bars = 'no', print_value_counts=False):
    """Esta función entrega un gráfico de barras con las frecuencias absolutas ordenadas de mayor a menor.

    Parameters
    ----------
    dataframe : [pd.DataFrame]
        DataFrame con la variable a graficar.
    
    columna : [str]
        Nombre de la variable a graficar

    n_bars : [int]
        Por defecto: 'no'. Permite elegir el número de clases de la variable que se van a grafícar.

    print_value_counts : [bool]
        Por defecto True. Imprime en pantalla las cifras de las frecuencias absolutas.

              
    Returns
    -------
    None
        
    """
    if n_bars == 'no':
        dataframe[columna].value_counts().sort_values().plot(kind='barh', color='lightblue')
    else:
        dataframe[columna].value_counts().head(n_bars).sort_values().plot(kind='barh', color='lightblue')
    plt.title(f'Frecuencia de las clases de la variable {dataframe[columna].name}')
    plt.axvline(dataframe[columna].value_counts().mean(), color='red', ls ='--', label = f'Media: {round(dataframe[columna].value_counts().mean(),2)}')
    plt.legend(fontsize = 15);
    plt.grid()
    if print_value_counts == True:
        print(dataframe[columna].value_counts())


def barplot_bivariado(dataframe, vo, vd_1, orient='vertical'):
    """Esta función entrega un gráfico de barras bivariado.

    Parameters
    ----------
    dataframe : [pd.DataFrame]
        DataFrame con las variables a graficar.
    
    vo : [str]
        Nombre de la variable dependiente
    
    vd_1 : [str]
        Nombre de la variable independiente

    orient : [str]
        Por defecto: 'vertical. Permite seleccionar la orientación del gráfico
              
    Returns
    -------
    None
        
    """
    tmp_order_list = dataframe.groupby(vd_1)[vo].mean().to_frame().sort_values(by=vo, ascending=False).index        
    if orient == 'vertical':
        sns.barplot(x=vd_1, y=vo, data=dataframe, color='lightblue', orient='v', order=tmp_order_list)
        plt.grid()
    if orient == 'horizontal':
        sns.barplot(x=vo, y=vd_1, data=dataframe, color='lightblue', orient='h', order=tmp_order_list)
        plt.grid()


def pointplot_bivariado(dataframe, vo, vd_1):
    """Esta función entrega un gráfico de puntos bivariado.

    Parameters
    ----------
    dataframe : [pd.DataFrame]
        DataFrame con las variables a graficar.
    
    vo : [str]
        Nombre de la variable dependiente
    
    vd_1 : [str]
        Nombre de la variable independiente
              
    Returns
    -------
    None
        
    """
    #sns.pointplot(x=dataframe[vo], y=dataframe[vd_1], join=False, order=tmp_order_list)
    plt.title(f'{dataframe[vo].name} para {dataframe[vd_1].name}')
    plt.xlabel(f'{vo}');



def pointplot_multivariado(dataframe, vo, vd_1, vd_2):
    """Esta función entrega un gráfico de puntos multivariado.

    Parameters
    ----------
    dataframe : [pd.DataFrame]
        DataFrame con las variables a graficar.
    
    vo : [str]
        Nombre de la variable dependiente
    
    vd_1 : [str]
        Nombre de la primera variable independiente

    vd_1 : [str]
        Nombre de la segunda variable independiente
              
    Returns
    -------
    None
        
    """
    sns.pointplot(x=dataframe[vo], y=dataframe[vd_1], hue=dataframe[vd_2], join=False)
    plt.title(f'{dataframe[vo].name} para {dataframe[vd_1].name} y {dataframe[vd_2].name}')
    plt.xlabel(f'{vo}');



######################################DATOS PERDIDOS#####################################################
def imputer_preprocessing(dataframe, columna):
    """Esta función prepara los datos para la imputación transformando en dato nulo los registros de todas las columnas resultantes de la binarización cuya columna de origen contenía un dato perdido.

    Parameters
    ----------
    dataframe : [pd.DataFrame]
        DataFrame binarizado que debe incluir binarizadas también las columnas de los datos nulos de cada variable.
    
    columna : [str]
        Nombre de la variable que queremos procesar
           

    Returns
    -------
    [pd.DataFrame]
        Retorna un DataFrame con datos datos nulos para todas las variables binarizadas cuya variable de origen tenia un dato perdido.
        
    """

    for rowname, rowseries in dataframe[dataframe[f'{columna}_nan']==1].iterrows():
        for bin_col in dataframe.filter(regex = columna).columns:
            dataframe.loc[rowname, bin_col] = np.nan
    return dataframe

######################################MODELAMIENTO PREDICTIVO#####################################################
def estandarizacion(lista_columnas, X_train, X_test):
    """Esta función realiza la estandarizacion de una lista de variable continuas excluyendo de la operación las variables binarias.

    Parameters
    ----------
    lista_columnas : [list]
        Lista con el nombre de las variables continuas a estandarizar.
    
    X_train : [pd.DataFrame]
        Dataframe de entrenamiento donde estan contenidas las variables a estandarizar.

    X_test : [pd.DataFrame]
        Dataframe de prueba donde estan contenidas las variables a estandarizar.
           

    Returns
    -------
    X_train_std : [pd.DataFrame]
        Dataframe de entrenamiento con las variables numéricas estandarizadas.

    X_test_std : [pd.DataFrame]
        Dataframe de test con las variables numéricas estandarizadas.

    scaler : [instancia de la clase StandardScaler]
        Parámetros de la estandarización.
        
    """

    scaler = StandardScaler()  
    # Almacemanos los nombres de las columnas estandarizables en un objeto
    columnas_estandarizables = lista_columnas

    # Entrenamos sobre las columnas estandarizables de la muestra de entrenamiento
    scaler.fit(X_train[columnas_estandarizables])

    # Generamos copias de los subconjuntos de atributos donde sobreescribiremos la transformación
    X_train_std = X_train.copy()
    X_test_std = X_test.copy()

    # Transformamos
    X_train_std[columnas_estandarizables] = scaler.transform(X_train[columnas_estandarizables])
    X_test_std[columnas_estandarizables] = scaler.transform(X_test[columnas_estandarizables])

    return X_train_std, X_test_std, scaler


def entrenamiento_modelo_vanilla(model, X_train, y_train):
    """Esta función entrena un modelo sin ajuste de hiperparámetros.

    Parameters
    ----------
    model : [model]
        Modelo predictivo instanciado.
    
    X_train : [pd.DataFrame]
        Dataframe con la matriz de atributos independientes de entrenamiento.

    y_train : [pd.DataFrame]
        Vector objetivo de entrenamiento.

    Returns
    -------
    model : [model]
        Retorna el modelo entrenado.
        
    """
    model.fit(X_train, y_train)
    return model


def serializar_modelos(model_trained, iteration_name):
    """Esta función serializa un modelo entrenado como un archivo .pkl y le asigna un nombre.

    Parameters
    ----------
    model_trained : [model]
        Modelo predictivo entrenado.
    
    iteration_name : [str]
        Sufijo que se añadirá al final del nombre de la clase del modelo entrenado.


    Returns
    -------
    None
        
    """
    # extraemos el nombre del modelo
    tmp_model_name = str(model_trained.__class__).replace("'>", '').split('.')[-1]
    # generamos un timestamp
    #tmp_time_stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    # generamos el pickle, cada modelo entrenado tendrá el nombre del modelo y la hora a la que se generó
    pickle.dump(model_trained, open(f"./{tmp_model_name}_{iteration_name}.pkl", 'wb'))
    # Imprimimos un mensaje de confirmación
    print(f"El archivo {tmp_model_name}_{iteration_name}.pkl fue generado")


def predicciones_metricas(pickle_name, X_test, y_test, print_results=False):
    """Esta función recibe un modelo de regresión entrenado y almacenado como archivo .pkl y calcula sus métricas de rendimiento. Está adaptado para un vector objetivo con transformación logarítmica

    Parameters
    ----------
    pickle_name : [model]
        Modelo predictivo entrenado almacenado como archivo .pkl.
    
    X_test : [pd.DataFrame]
        Dataframe con la matriz de atributos independientes de prueba.

    y_test : [pd.DataFrame]
        Vector objetivo de prueba.

    Returns
    -------
    r2_tmp : [float]
        R2.

    rmse_tmp : [float]
        RMSE.

    mae_tmp : [float]
        MAE.

    mape_tmp : [float]
        MAPE.
        
    """
    # importamos los pickles
    read_model_tmp = pickle.load(open(pickle_name, 'rb'))
    
    # generamos predicciones
    y_hat_tmp = read_model_tmp.predict(X_test)
    
    # obtenemos métricas
    r2_tmp = round(r2_score(np.exp(y_test), np.exp(y_hat_tmp)), 2)
    rmse_tmp = round(np.sqrt(mean_squared_error(np.exp(y_test), np.exp(y_hat_tmp))), 2)
    mae_tmp = round(median_absolute_error(np.exp(y_test), np.exp(y_hat_tmp)),2)
    mape_tmp = round(mean_absolute_percentage_error(np.exp(y_test), np.exp(y_hat_tmp)),2)
    
    # imprimimos resultados
    if print_results is True:
        print(f'R2: {r2_tmp}')
        print(f'RMSE: {rmse_tmp}')
        print(f'MAE: {mae_tmp}')
        print(f'MAPE: {mape_tmp}')
    
    return r2_tmp, rmse_tmp, mae_tmp, mape_tmp



def plot_importance(fit_model, feat_names):
    """Esta función imprime un gráfico con las importancias de los atributos de un modelo basado en árboles de decisión

    Parameters
    ----------
    fit_model : [model]
        Modelo predictivo entrenado.
    
    feat_names : [list]
        Lista con los nombres de los atributos. Se puede obtener como X_mat.columns.

    Returns
    -------
    None
        
    """
    tmp_importance = fit_model.feature_importances_
    sort_importance = np.argsort(tmp_importance)[::-1]
    names = [feat_names[i] for i in sort_importance]
    plt.title("Feature importance")
    plt.barh(range(len(feat_names)), tmp_importance[sort_importance])
    plt.yticks(range(len(feat_names)), names, rotation=0)