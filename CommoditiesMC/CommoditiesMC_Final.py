# Necessary libraries
import random
import pandas as pd
import yaml
import sys
import os
import time
from statistics import mean
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
import numpy as np


# Check if the user provided a path for the User.yaml file
if len(sys.argv) > 1:
    user_path = sys.argv[1]
else:
    sys.exit("User.yaml path not provided.")

# Relative file path to be used
current_directory = os.path.dirname(__file__)

# YAML file with user instructions and consuptions database
database_path = os.path.join(current_directory, 'Database.yaml')
with open(user_path, "r") as user_yaml:
    User = yaml.safe_load(user_yaml)
with open(database_path, "r") as database_yaml:
    Database = yaml.safe_load(database_yaml)

# Read Forecast
matching_files = [filename for filename in os.listdir(current_directory) if os.path.isfile(os.path.join(current_directory, filename))]
forecast_file = next((file for file in matching_files if file.lower().endswith('.xlsx')), None)

if forecast_file:
    df_energies_template = pd.read_excel(os.path.join(current_directory, forecast_file))
elif os.path.isfile(User['Path_Database']):
    df_energies_template = pd.read_excel(User['Path_Database'])
else:
    csv_file = next((file for file in matching_files if file.lower().endswith('.csv')), None)
    if csv_file:
        csv_path = os.path.join(current_directory, csv_file)
        xlsx_path = os.path.join(current_directory, os.path.splitext(csv_file)[0] + '.xlsx')
        # Convert CSV to XLSX and read as DataFrame
        pd.read_csv(csv_path).to_excel(xlsx_path, index=False)
        df_energies_template = pd.read_excel(xlsx_path)
    else:
        sys.exit("No suitable data file found.")
def plot_selected_columns_script(analysis_df, selected_columns, output_dir):
    """
    Plots the selected columns with their mean and standard deviation.

    Args:
        analysis_df (pd.DataFrame): DataFrame with the average and standard deviation for each day.
        selected_columns (list): List of columns to plot.
        output_dir (str): Directory where the plots will be saved.
    """
    plt.figure(figsize=(12, 6))

    for col in selected_columns:
        mean_col = f"{col}_mean"
        std_col = f"{col}_std"
        if mean_col in analysis_df.columns and std_col in analysis_df.columns:
            plt.plot(analysis_df.index, analysis_df[mean_col], label=f'{col} Mean')
            plt.fill_between(
                analysis_df.index,
                analysis_df[mean_col] - analysis_df[std_col],
                analysis_df[mean_col] + analysis_df[std_col],
                alpha=0.2,
                label=f'{col} Std Dev'
            )

    plt.title('Selected Columns over Time')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_filename = '_'.join(selected_columns) + '_plot.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path)
    plt.show()
    plt.close()

    print(f"Plot saved to {plot_path}")

def interactive_plot_script(analysis_df, output_dir):
    """
    Creates an interactive plot menu to select columns for plotting using numerical input.

    Args:
        analysis_df (pd.DataFrame): DataFrame with the average and standard deviation for each day.
        output_dir (str): Directory where the plots will be saved.
    """
    # Extract columns for plotting
    mean_columns = [col for col in analysis_df.columns if col.endswith('_mean')]
    base_columns = [col.replace('_mean', '') for col in mean_columns]

    while True:
        # Display available columns with numbers
        print("\nAvailable columns:")
        for idx, col in enumerate(base_columns, 1):
            print(f"{idx}. {col}")

        # Ask the user to select columns by entering numbers
        selected_numbers = input("Enter the numbers of columns to plot, separated by commas (or type 'exit' to finish): ").strip()
        if selected_numbers.lower() == 'exit':
            break

        selected_indices = [int(num.strip()) - 1 for num in selected_numbers.split(',') if num.strip().isdigit()]

        selected_columns = [base_columns[idx] for idx in selected_indices if 0 <= idx < len(base_columns)]

        if selected_columns:
            plot_selected_columns_script(analysis_df, selected_columns, output_dir)
        else:
            print("Invalid selection. Please try again.")

# Function to obtain the difference between Production minus Demand
def surplus(df_energies):
    """Creates a dataframe with Date and the difference between
    demand and total produced, showing electrical surplus or deficit in kW for each date"""
    df = pd.DataFrame()
    df['Date'] = df_energies['ds']

    data_unit = User['Database_Unit'].upper()
    if data_unit == 'GW':
        unit = 1000000
    elif data_unit == 'MW':
        unit = 1000
    elif data_unit == 'KW':
        unit = 1
    else:
        sys.exit("Invalid database unit in User.yaml")

    df['Prod_Dem_Diff (kW)'] = (df_energies['Total'] - df_energies['Demand']) * unit
    df['Accum_Diff (kW)'] = df['Prod_Dem_Diff (kW)'].cumsum()
    return df

# Function to collect technology consumption values
def technology_values():
    """Stores the names of production technologies selected
    in User.yaml as well as their power consumption"""
    tech_prod = []
    for value in User['Production_Share'].values():
        techs = []
        for item in value[1:]:
            if isinstance(item, str):
                techs.append(item)
            elif isinstance(item, dict):
                techs.extend(item.values())
        tech_prod.append(techs)

    values = []
    for tech_list in tech_prod:
        tech_values = []
        for tech in tech_list:
            for key1, value1 in Database['Electricity_Consumption'].items():
                for key2, value2 in value1.items():
                    if tech == key2:
                        ran = User['Random_Value']
                        if isinstance(value2, list) and len(value2) == 2 and ran:
                            tech_values.append(round(random.uniform(value2[0], value2[1]), 4))
                        else:
                            tech_values.append(mean(value2))
        values.append(tech_values)

    result = []
    for sublist in values:
        nested = []
        for item in sublist:
            if isinstance(item, list):
                nested.extend(item)
            else:
                nested.append(item)
        result.append(nested)

    single_comm = []
    dependent_comm = []
    for item in result:
        if len(item) == 1:
            single_comm.append(item[0])
        else:
            dependent_comm.append(item)

    return single_comm, dependent_comm

# Function to collect production share
def production_share():
    """Stores the percentages described in User.yaml"""
    single_perc = []
    dependent_perc = []

    for _, perc in User['Production_Share'].items():
        if len(perc) == 2:
            single_perc.append(float(perc[0]))
        else:
            for value in perc:
                if isinstance(value, (float, int)):
                    dependent_perc.append(float(value))
    if sum(single_perc + dependent_perc) == 1.0:
        return single_perc, dependent_perc
    else:
        print("The sum of percentages must equal 1. Please check User.yaml.")
        sys.exit()

# Function to generate production of commodities
def final_production(df_energies):
    sur = surplus(df_energies)

    single_cons, dependent_cons = technology_values()

    single_perc, dependent_perc = production_share()

    dep_values = []
    for sust in User.get('Production_Share'):
        if sust in Database.get('Mass_Balance'):
            metric_val = Database['Mass_Balance'][sust]
            temp_list = []
            for row in metric_val.values():
                ran = User['Random_Value']
                if isinstance(row, list) and len(row) == 2 and ran:
                    random_value = round(random.uniform(row[0], row[1]), 4)
                    temp_list.append(random_value)
                else:
                    temp_list.append(mean(row))
            dep_values.append(temp_list)

    single_names = []
    dependent_names = []
    for key, value in User['Production_Share'].items():
        if len(value) == 2:
            single_names.append(key)
        elif len(value) > 2:
            dependent_names.append(key)

    sum_val = []
    for sublist in dependent_cons:
        sum_val.append(sublist.pop(0))
    result_list = []
    for i in range(len(dep_values)):
        result = 0
        for j in range(len(dep_values[i])):
            result += dep_values[i][j] * dependent_cons[i][j]
        result += sum_val[i]
        result_list.append(result)

    for col in single_names + dependent_names:
        sur[col] = 0.0
        sur[col]=sur[col].astype('float64')
    for index, row in sur.iterrows():
        if row['Prod_Dem_Diff (kW)'] > 0:
            for i, col in enumerate(single_names):
                if col in single_names:
                    sur.at[index, col] = row['Prod_Dem_Diff (kW)'] * single_perc[i] / single_cons[i] if single_cons[i] != 0 else 0
            for i, col in enumerate(dependent_names):
                if col in dependent_names:
                    sur.at[index, col] = row['Prod_Dem_Diff (kW)'] * dependent_perc[i] / result_list[i]
    return sur

# Function to transform commodities produced into electrical energy
def to_electricity(df_energies):
    surplus = final_production(df_energies)

    result_dict = {}
    names = surplus.columns.tolist()
    for names in Database['Commodities_to_Electricity']:
        values = Database['Commodities_to_Electricity'][names]
        for key, val in values.items():
            ran = User['Random_Value']
            efficiency_values = val[0]
            energy_density_values = val[1]
            if len(efficiency_values) == 2 and ran:
                efficiency = random.uniform(*efficiency_values)
            else:
                efficiency = mean(efficiency_values)
            if len(energy_density_values) == 2 and ran:
                energy_density = random.uniform(*energy_density_values)
            else:
                energy_density = mean(energy_density_values)

            result = round(efficiency * energy_density, 4)
        result_dict[names] = result

    fuels = User['Fuels_Burning']
    for key, perc in fuels.items():
        if key in result_dict and perc <= 1.0 and perc > 0.0:
            col_name_accum = f'Accum_{key}'
            surplus[col_name_accum] = surplus[key].cumsum()
            col_name = f'{key} (kW)'
            surplus[col_name] = surplus[key] * perc * result_dict[key]

    columns_to_sum = surplus.filter(regex='(kW)').columns.difference(
        surplus.filter(like='Prod_Dem_Diff').columns).difference(
        surplus.filter(like='Accum_Diff').columns)
    surplus['Total_Potential (kW)'] = surplus[columns_to_sum].sum(axis=1)
    surplus['Accum_Total_Potential (kW)'] = surplus['Total_Potential (kW)'].cumsum()
    surplus['Accum_Burned_Potential (kW)'] = surplus['Total_Potential (kW)'].cumsum()
    surplus['Burned_Diff (kW)'] = surplus['Prod_Dem_Diff (kW)']

    reset_accum = False
    for index, row in surplus.iterrows():
        if index == 0:
            continue
        if row['Prod_Dem_Diff (kW)'] < 0:
            if reset_accum:
                surplus.at[index, 'Accum_Burned_Potential (kW)'] = 0
                surplus.at[index, 'Burned_Diff (kW)'] = surplus.at[index, 'Accum_Burned_Potential (kW)'] + row[
                    'Prod_Dem_Diff (kW)']
            else:
                surplus.at[index, 'Accum_Burned_Potential (kW)'] = surplus.at[index - 1, 'Accum_Burned_Potential (kW)'] + \
                                                                    row['Prod_Dem_Diff (kW)']
            if surplus.at[index, 'Accum_Burned_Potential (kW)'] < 0:
                surplus.at[index, 'Burned_Diff (kW)'] = surplus.at[index, 'Accum_Burned_Potential (kW)']
                surplus.at[index, 'Accum_Burned_Potential (kW)'] = 0
                reset_accum = True
            else:
                reset_accum = False
        else:
            if reset_accum:
                surplus.at[index, 'Burned_Diff (kW)'] = surplus.at[index, 'Accum_Burned_Potential (kW)'] + row[
                    'Prod_Dem_Diff (kW)']
                surplus.at[index, 'Accum_Burned_Potential (kW)'] = row['Total_Potential (kW)']
                reset_accum = False
            else:
                surplus.at[index, 'Accum_Burned_Potential (kW)'] = surplus.at[index - 1, 'Accum_Burned_Potential (kW)'] + \
                                                                    row['Total_Potential (kW)']

    for index, row in surplus.iterrows():
        if index == 0:
            continue
        if row['Prod_Dem_Diff (kW)'] < 0 and row['Accum_Burned_Potential (kW)'] > 0:
            surplus.at[index, 'Burned_Diff (kW)'] = 0
        if row['Prod_Dem_Diff (kW)'] > 0 and surplus.at[index - 1, 'Accum_Burned_Potential (kW)'] == 0:
            surplus.at[index, 'Burned_Diff (kW)'] = row['Prod_Dem_Diff (kW)']

    surplus['Accum_Burned_Diff (kW)'] = surplus['Burned_Diff (kW)'].cumsum()
    surplus['Covered_Deficit (kW)'] = surplus['Prod_Dem_Diff (kW)'] - surplus['Burned_Diff (kW)']
    surplus['Deficit_to_Cover(kW)'] = surplus['Burned_Diff (kW)'].apply(lambda x: x if x < 0 else 0)
    surplus['Accum_Def_to_Cover(kW)'] = surplus['Deficit_to_Cover(kW)'].cumsum()

    return surplus

# Function to export data to an Excel file
def excel(df_energies,j):
    sur = final_production(df_energies)
    surplus = to_electricity(df_energies)

    sur['Date'] = pd.to_datetime(sur['Date']).dt.date
    surplus['Date'] = pd.to_datetime(sur['Date']).dt.date

    elec_unit = User['Electricity_Output'].upper()
    if elec_unit == 'GW':
        unit = 1000000
    elif elec_unit == 'MW':
        unit = 1000
    elif elec_unit == 'KW':
        unit = 1
    else:
        sys.exit("Invalid electricity unit in User.yaml")

    columns_sur = [col for col in sur.columns if 'kW' in col]
    columns_surplus = [col for col in surplus.columns if 'kW' in col]

    sur[columns_sur] = sur[columns_sur].div(unit)
    surplus[columns_surplus] = surplus[columns_surplus].div(unit)

    sur.rename(columns=lambda x: x.replace('kW', elec_unit), inplace=True)
    surplus.rename(columns=lambda x: x.replace('kW', elec_unit), inplace=True)

    for df in [sur, surplus]:
        for column in df.columns:
            if elec_unit not in column:
                if column != 'H2O' and column != 'Date':
                    df.rename(columns={column: f"{column} (Kg)"}, inplace=True)
                elif column == 'H2O':
                    df.rename(columns={column: f"{column} (Lt)"}, inplace=True)

    file_names = ['Commodities_Production_{}.xlsx'.format(j), 'Commodities_to_Electricity_{}.xlsx'.format(j)]
    for i, df in enumerate([sur, surplus]):
        path = os.path.join(current_directory, 'Output', file_names[i])
        df.to_excel(path, index=False)

    print("Simulation completed successfully")

# Function to perform Monte Carlo simulation
def monte_carlo_simulation(num_iterations):
    forecast_df = df_energies_template
    desviaciones_estandar = pd.read_csv('desviaciones_estandar.csv')
    alls_mean=pd.read_csv('all_mean.csv')
    for i in range(num_iterations):
        start_time = time.time()
        # Generate perturbed df_energies DataFrame
        perturbed_df_energies = perturb_df_energies(forecast_df,desviaciones_estandar,alls_mean)
        print(f"Iteration {i + 1}:")
        excel(perturbed_df_energies,i)
        print()
        end_time = time.time()  # Detiene el temporizador
        elapsed_time = end_time - start_time
        print(f"Elapsed time for iteration {i + 1}: {elapsed_time:.2f} seconds\n")
    
    output_directory = os.path.join(current_directory, 'Output')
    start_time = time.time()
    analyze_monte_carlo_results(num_iterations, output_directory)
    end_time = time.time()  # Detiene el temporizador
    elapsed_time = end_time - start_time
    print(f"Elapsed time for analyze_monte_carlo_results: {elapsed_time:.2f} seconds\n")
    analysis_file = os.path.join(output_directory, 'Monte_Carlo_Analysis.xlsx')
    analysis_df = pd.read_excel(analysis_file, index_col='Date', parse_dates=True)
    interactive_plot_script(analysis_df, output_directory)


def analyze_monte_carlo_results(num_iterations, output_dir):
    """
    Analyzes Monte Carlo simulation results by calculating the average and standard deviation for each day
    across all simulations.

    Args:
        num_iterations (int): Number of Monte Carlo iterations.
        output_dir (str): Directory where the simulation results are stored.

    Returns:
        pd.DataFrame: DataFrame with the average and standard deviation for each day.
    """
    # List to store dataframes from each simulation
    simulation_results = []

    # Load each simulation result
    for i in range(num_iterations):
        file_path = os.path.join(output_dir, f'Commodities_to_Electricity_{i}.xlsx')
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
            simulation_results.append(df)
        else:
            print(f"File {file_path} does not exist. Skipping iteration {i}.")

    if not simulation_results:
        print("No simulation results found. Exiting.")
        return None

    # Concatenate all dataframes along a new axis
    combined_df = pd.concat(simulation_results, keys=range(num_iterations), names=['Iteration', 'Index'])

    # Group by date and calculate mean and standard deviation
    grouped = combined_df.groupby('Date')
    means = grouped.mean()
    stds = grouped.std()

    # Combine means and standard deviations into one dataframe
    result_df = pd.concat([means.add_suffix('_mean'), stds.add_suffix('_std')], axis=1)

    # Export to a new Excel file
    output_file = os.path.join(output_dir, 'Monte_Carlo_Analysis.xlsx')
    result_df.to_excel(output_file)
    
    print(f"Monte Carlo analysis completed. Results saved to {output_file}")
    return result_df


def perturb_df_energies(forecast_df,desviaciones_estandar,alls_mean):
    

    # Procesar archivos .CSV
    desviaciones_estandar_dic=dict(zip(desviaciones_estandar['Tecnología'],desviaciones_estandar['Desviación estándar']))
    alls_mean_dic=dict(zip(alls_mean['Tecnología'],alls_mean['Media']))



    # Example of perturbation: Generating normally distributed values around original df_energies
    perturbed_forecast_df=forecast_df.copy()
    sigma=2
    # Identificar columnas de tecnologías en forecast_df que coinciden con desviaciones_estandar
    tecnologias = [col for col in forecast_df.columns if col in desviaciones_estandar_dic and 'trend' not in col.lower()]
    
    
    # Perturbar los valores de las tecnologías, incluyendo Demand
    for tecnologia in tecnologias:
        mu=alls_mean_dic[tecnologia]
        std = desviaciones_estandar_dic[tecnologia]
        perturbed_forecast_df[tecnologia] = perturbed_forecast_df[tecnologia].apply(lambda x: max(0, x+((np.random.uniform(-1,1)*((std*sigma)/mu))*x)))

    # Excluir 'Nuclear_Centrals_trend' y 'Demand' solo para el cálculo del Total
    tecnologias_para_total = [col for col in tecnologias if col not in ['Nuclear_Centrals_trend', 'Demand']]

    # Recalcular la columna Total
    if 'Total' in perturbed_forecast_df.columns:
        perturbed_forecast_df['Total'] = perturbed_forecast_df[tecnologias_para_total].sum(axis=1)
    return perturbed_forecast_df

if __name__ == "__main__":
    # Number of Monte Carlo iterations
    num_iterations = User['Number_of_simulation']
    print(f"Running {num_iterations} iterations of Monte Carlo simulation...\n")
    monte_carlo_simulation(num_iterations)
