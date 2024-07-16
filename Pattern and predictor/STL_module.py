import pandas as pd
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt

def calculate_std_with_stl(df, column_name):
    try:
        stl = STL(df[column_name].astype(float), seasonal=13)  # Ajusta el parámetro seasonal según la estacionalidad esperada
        result = stl.fit()
        residuo = result.resid
        desviacion_estandar = residuo.std()
        print(f"Desviación estándar de {column_name}: {desviacion_estandar}")
        return desviacion_estandar
    except ValueError as e:
        print(f"Error processing column {column_name}: {e}")
        return None

def calculate_all_means(prod_df, demand_df, output_file):
    all_mean = {}

    for columna in prod_df.columns:
        if pd.api.types.is_numeric_dtype(prod_df[columna]):
            mean_tecnologia = prod_df[columna].mean()
            if mean_tecnologia is not None:
                all_mean[columna] = mean_tecnologia

    for columna in demand_df.columns:
        if pd.api.types.is_numeric_dtype(demand_df[columna]):
            mean_tecnologia = demand_df[columna].mean()
            if mean_tecnologia is not None:
                all_mean[columna] = mean_tecnologia

    df_all_mean_tecnologia = pd.DataFrame(all_mean.items(), columns=['Tecnología', 'Media'])
    df_all_mean_tecnologia.to_csv(output_file, index=False)
    return df_all_mean_tecnologia

def calculate_all_stds(prod_df, demand_df, output_file):
    desviaciones_estandar = {}

    for columna in prod_df.columns:
        if pd.api.types.is_numeric_dtype(prod_df[columna]):
            desviacion_estandar = calculate_std_with_stl(prod_df, columna)
            if desviacion_estandar is not None:
                desviaciones_estandar[columna] = desviacion_estandar

    for columna in demand_df.columns:
        if pd.api.types.is_numeric_dtype(demand_df[columna]):
            desviacion_estandar = calculate_std_with_stl(demand_df, columna)
            if desviacion_estandar is not None:
                desviaciones_estandar[columna] = desviacion_estandar

    df_desviaciones_estandar = pd.DataFrame(desviaciones_estandar.items(), columns=['Tecnología', 'Desviación estándar'])
    df_desviaciones_estandar.to_csv(output_file, index=False)
    return df_desviaciones_estandar

def plot_mu(df_all_mu, output_image):
    plt.figure(figsize=(10, 6))
    for index, row in df_all_mu.iterrows():
        plt.plot(row['Tecnología'], row['Media'], 'o', label=row['Tecnología'])
    plt.xlabel('Tecnología')
    plt.ylabel('Promedio')
    plt.title('Promedio por Tenologia y demanda')
    plt.legend()
    plt.savefig(output_image)
    plt.close()
    
def plot_stds(df_desviaciones_estandar, output_image):
    plt.figure(figsize=(10, 6))
    for index, row in df_desviaciones_estandar.iterrows():
        plt.plot(row['Tecnología'], row['Desviación estándar'], 'o', label=row['Tecnología'])
    plt.xlabel('Tecnología')
    plt.ylabel('Desviación estándar')
    plt.title('Desviación estándar resultante de la descomposición STL')
    plt.legend()
    plt.savefig(output_image)
    plt.close()