import streamlit as st
import pickle
import pandas as pd
import numpy as np


# Configurar las características generales de la página
st.set_page_config(
    page_title="MotorPricingGermany",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="auto",
)



# Título y encabezado de la aplicación
st.title("Motor Pricing Germany")
st.subheader('Estimación de precios de vehículos usados con Machine Learning')

texto = """
<blockquote style="background-color:#f0f0f0; padding:10px; line-height: 1;">
<p style="font-size:14px;">Instrucciones:</p>
<p style="font-size:12px; margin-left: 30px;">&#8226; Ingresa los datos solicitados en el panel lateral.</p>
<p style="font-size:12px; margin-left: 30px;">&#8226; Presiona el botón 'Calcular precio'.</p>
<p style="font-size:12px; margin-left: 30px;">&#8226; Los resultados se mostrarán en la parte central de la página.</p>

<br> <!-- Salto de línea para crear espacio vertical -->

<p style="font-size:14px;">Limitaciones:</p>
<p style="font-size:12px; margin-left: 30px;">&#8226; Los datos de entrenamiento pertenecen al año 2016 en Alemania.</p>
</blockquote>
"""

# Mostrar el texto utilizando markdown
st.markdown(texto, unsafe_allow_html=True)



st.markdown("""***""")

# Barra lateral

# Ruta relativa a la imagen en la misma carpeta que el archivo .py
ruta_imagen = "img.png"

# Mostrar la imagen en el panel lateral
st.sidebar.image(ruta_imagen, use_column_width=True)



st.sidebar.header('Introduce los datos del vehículo')



# ENTRADA DE DATOS
# Lista de opciones para las variables categóricas

opciones_brand = {
    '': '',
    'Jeep': 'jeep',
    'Volkswagen': 'volkswagen',
    'Skoda': 'skoda',
    'Peugeot': 'peugeot',
    'Ford': 'ford',
    'Mazda': 'mazda',
    'Renault': 'renault',
    'Mercedes': 'mercedes',
    'BMW': 'bmw',
    'Seat': 'seat',
    'Honda': 'honda',
    'Fiat': 'fiat',
    'Mini': 'mini',
    'Smart': 'smart',
    'Audi': 'audi',
    'Nissan': 'nissan',
    'Opel': 'opel',
    'Alfa Romeo': 'alfa',
    'Subaru': 'subaru',
    'Volvo': 'volvo',
    'Mitsubishi': 'mitsubishi',
    'Hyundai': 'hyundai',
    'Lancia': 'lancia',
    'Citroen': 'citroen',
    'Toyota': 'toyota',
    'Kia': 'kia',
    'Chevrolet': 'chevrolet',
    'Dacia': 'dacia',
    'Daihatsu': 'daihatsu',
    'Suzuki': 'suzuki',
    'Chrysler': 'chrysler',
    'Dodge': 'dodge',
    'Daewoo': 'daewoo',
    'Rover': 'rover',
    'Saab': 'saab',
    'Lexus': 'lexus',
    'Abarth': 'abarth',
    'MG': 'mg',
    'Jaguar': 'jaguar',
    'Land Rover': 'land',
    'Lada': 'lada',
    'Iveco': 'iveco',
    'Ssangyong': 'ssangyong'
}
opciones_vehicleType = {
    '': '',
    'SUV': 'suv',
    'Compacto': 'kleinwagen',
    'Convertible': 'cabrio',
    'Monovolumen/Furgón': 'bus',
    'Sedán': 'limousine',
    'Familiar': 'kombi',
    'Coupé': 'coupe'
}
opciones_gearbox = {
    '': '',
    'Manual': 'manuell',
    'Automático': 'automatik'
}
opciones_fuelType = {
    '': '',
    'Diesel': 'diesel',
    'Gasolina': 'benzin',
    'GLP/GNC': 'gas',
    'Híbrido/eléctrico': 'hybrid'
}
opciones_postalZone = ['', '9', '6', '2', '5', '3', '8', '4', '7', '0', '1']


# Permitimos al usuario seleccionar las características del auto
seleccion_brand = st.sidebar.selectbox("Marca", list(opciones_brand.keys()))
seleccion_vehicleType = st.sidebar.selectbox("Tipo de carrocería", list(opciones_vehicleType.keys()))
seleccion_gearbox = st.sidebar.selectbox("Caja de cambios", list(opciones_gearbox.keys()))
seleccion_fuelType = st.sidebar.selectbox("Tipo de combustible", list(opciones_fuelType.keys()))
seleccion_postalZone = st.sidebar.selectbox("Zona Postal", opciones_postalZone)
seleccion_yearOfRegistration = st.sidebar.slider("Año de registro", min_value=1995, max_value=2015)
seleccion_powerPS = st.sidebar.slider("Caballos de potencia", min_value=50, max_value=250, value=100, step=5)
seleccion_kilometer = st.sidebar.slider("Kilómetros recorridos", min_value=5000, max_value=500000, value=100000, step=5000)

data_entrada = {'Marca': [seleccion_brand],
                'Tipo de carrocería': [seleccion_vehicleType],
                'Caja de cambios': [seleccion_gearbox],
                'Tipo de combustible': [seleccion_fuelType],
                'Zona Postal': [seleccion_postalZone],
                'Año de registro': [int(seleccion_yearOfRegistration)],
                'Caballos de potencia': [seleccion_powerPS],
                'Kilómetros recorridos': [seleccion_kilometer]
                }

caracteristicas_vehiculo = pd.DataFrame(data_entrada).T


# TRANSFORMACIÓN DE LOS DATOS DE ENTRADA
# Cargamos las columnas codificadas con la misma estructura que usamos en el entrenamiento en forma de dataframe vacío
columns_encoded = pd.read_csv('columns_encoded.csv')
X_user = columns_encoded.copy()

# Transformamos los datos continuos con el mismo objeto scaler que utilizamos en el entrenamiento
scaler = pickle.load(open('scaler_imp.pkl', 'rb'))
input_features_num = [[seleccion_yearOfRegistration, seleccion_powerPS, seleccion_kilometer]]
input_features_std = scaler.transform(input_features_num)

# Introducimos en el dataframe de predicciones los datos continuos transformados 
X_user.loc[0, 'yearOfRegistration'] = input_features_std[0][0]
X_user.loc[0, 'powerPS'] = input_features_std[0][1]
X_user.loc[0, 'kilometer'] = input_features_std[0][2]

# Codificamos las opciones categóricas seleccionadas por el usuario adaptandolas a la estructura de los datos de entrenamiento binarizados
for col in X_user.filter(regex = 'brand').columns:
    if opciones_brand[seleccion_brand] in col:
        X_user.loc[0, col] = 1
    else:
        X_user.loc[0, col] = 0
        
for col in X_user.filter(regex = 'vehicleType').columns:
    if opciones_vehicleType[seleccion_vehicleType] in col:
        X_user.loc[0, col] = 1
    else:
        X_user.loc[0, col] = 0
        
for col in X_user.filter(regex = 'gearbox').columns:
    if opciones_gearbox[seleccion_gearbox] in col:
        X_user.loc[0, col] = 1
    else:
        X_user.loc[0, col] = 0

for col in X_user.filter(regex = 'fuelType').columns:
    if opciones_fuelType[seleccion_fuelType] in col:
        X_user.loc[0, col] = 1
    else:
        X_user.loc[0, col] = 0
        
for col in X_user.filter(regex = 'postalZone').columns:
    if seleccion_postalZone in col:
        X_user.loc[0, col] = 1
    else:
        X_user.loc[0, col] = 0


X_user = X_user.astype('float64')



# OBTENCION DE PREDICCIONES
# Cargamos el modelo entrenado
model = pickle.load(open('LinearRegression_it1_imp.pkl', 'rb'))

# Generamos predicciones y deshacemos la transformación logarítmica del vector objetivo, redondeamos
y_pred = round(np.exp(model.predict(X_user)[0]))



# MOSTRAMOS RESULTADOS
# Botón
if st.button("Calcular precio"):
    # Validamos que se hayan rellenado correctamente los campos
    if not seleccion_brand or not seleccion_vehicleType or not seleccion_gearbox or not seleccion_gearbox or not seleccion_fuelType or not seleccion_postalZone or not seleccion_yearOfRegistration or not seleccion_powerPS or not seleccion_kilometer:
        st.error("Por favor, complete correctamente todos los campos.")
    # Llamamos a la función de predicción cuando el botón es presionado
    else:
        st.markdown('<span style="font-size: 20px;">Características del vehículo</span>', unsafe_allow_html=True)
        st.table(caracteristicas_vehiculo)
        contenedor_resultado = st.container()
        contenedor_resultado.markdown(f'<div style="background-color: #f5f5f5; padding: 20px; text-align: center;"><h2>Valor estimado</h2><p style="font-size: 24px; color: black;">{y_pred} euros</p></div>', unsafe_allow_html=True)
    

def footer_template():
    st.markdown(
        """
        <style>
        .footer {
            text-align: center;
            font-size: 14px;
            color: #808080;
            padding: 10px;
        }
        .footer a {
            color: #808080;
            margin: 0 10px;
            text-decoration: none;
        }
        .footer a:hover {
            color: #000000;
        }
        </style>
        <div class="footer">
            Daniel Almería &nbsp;&nbsp;&nbsp;
            <a href="mailto:daniel.alm92@gmail.com">Correo Electrónico</a> &nbsp;&nbsp;&nbsp;
            <a href="https://www.linkedin.com/in/daniel-almeria/" target="_blank">LinkedIn</a> &nbsp;&nbsp;&nbsp;
            <a href="https://www.github.com/DanielAL22" target="_blank">GitHub</a>
        </div>
        """,
        unsafe_allow_html=True
    )

# Llamada a la plantilla del pie de página
st.markdown("""***""")
footer_template()



















