#Necessary libraries
import random, pandas as pd, yaml, sys, os
from statistics import mean

# Check if the user provided a path for the User.yaml file
if len(sys.argv) > 1:
    user_path = sys.argv[1]

#Relative file path to be used
current_directory = os.path.dirname(__file__)

#YAML file with user instructions and consuptions database
database_path = os.path.join(current_directory, 'Database.yaml')
with open(user_path, "r") as user_yaml:
    User = yaml.safe_load(user_yaml)
with open(database_path, "r") as database_yaml:
    Database = yaml.safe_load(database_yaml)

#Read Forecast
matching_files = [filename for filename in os.listdir(current_directory) if os.path.isfile(os.path.join(current_directory, filename))]
forecast_file = next((file for file in matching_files if file.lower().endswith('.xlsx')), None)

if forecast_file:
    df_energies = pd.read_excel(os.path.join(current_directory, forecast_file))
elif os.path.isfile(User['Path_Database']):
    df_energies = pd.read_excel(User['Path_Database'])
else:
    csv_file = next((file for file in matching_files if file.lower().endswith('.csv')), None)
    if csv_file:
        csv_path = os.path.join(current_directory, csv_file)
        xlsx_path = os.path.join(current_directory, os.path.splitext(csv_file)[0] + '.xlsx')
        # Convert CSV to XLSX and read as DataFrame
        pd.read_csv(csv_path).to_excel(xlsx_path, index=False)
        df_energies = pd.read_excel(xlsx_path)

#--------------------------------------------------------------------------FUNCTIONS

print("Starting the simulation")
#Obtain the difference between Production minus Demand
def surplus():

    """A new dataframe is created that contains the date and the difference between
    the demand and the total produced, showing the electrical surplus or deficit in kW for that date"""
    df = pd.DataFrame()
    df['Date'] = df_energies['ds']

    "Unit of electricity selected according to the input"
    data_unit = User['Database_Unit'].upper()
    if data_unit == 'GW':
        unit = 1000000
    elif data_unit == 'MW':
        unit = 1000
    elif data_unit == 'KW':
        unit = 1
    else:
        sys.exit("Check User.yaml because database unit is invalid")

    df['Prod_Dem_Diff (kW)'] = (df_energies['Total'] - df_energies['Demand']) * unit
    df['Accum_Diff (kW)'] = df['Prod_Dem_Diff (kW)'].cumsum()
    return df

#Collect technology consumption values
def technology_values():

    """The names of the commidities production technologies selected
    in the User.yaml file are stored in the list named 'tech_prod'"""
    tech_prod = []
    for value in User['Production_Share'].values():
        techs = []
        for item in value[1:]:
            if isinstance(item, str):
                techs.append(item)
            elif isinstance(item, dict):
                techs.extend(item.values())
        tech_prod.append(techs)

    """The data from the tech_prod list is looked up in the Database.yaml file
    to obtain the power consumption and stored in the list called 'values'"""
    values = []
    for tech_list in tech_prod:
        tech_values = []
        for tech in tech_list:
            for key1, value1 in Database['Electricity_Consumption'].items():
                for key2, value2 in value1.items():
                    if tech == key2:
                        ran = User['Random_Value']
                        if isinstance(value2, list) and len(value2) == 2 and ran: 
                            tech_values.append(round(random.uniform(value2[0], value2[1]),4))#-----------------MODIFICAR A DISTRIBUCIÓN NORMAL
                        else:
                            tech_values.append(mean(value2))
        values.append(tech_values)

    "Nested list type values are eliminated to have a better later reading of the data"
    result = []
    for sublist in values:
        nested = []
        for item in sublist:
            if isinstance(item, list):
                nested.extend(item)
            else:
                nested.append(item)
        result.append(nested)

    "Generates two lists, independent and dependent production commodities"
    single_comm = []
    dependent_comm = []
    for item in result:
        if len(item) == 1:
            single_comm.append(item[0])
        else:
            dependent_comm.append(item)

    return single_comm, dependent_comm

#Collect production share
def production_share():

    "The percentages described in User.yaml are stored in a list"
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
        print("The sum of the percentages must equal 1. Enter percentages again")
        sys.exit()

#Generate the production of commodities
def final_production():

    "The surplus produced"
    sur = surplus()

    "Obtain two lists with value of consumption of commodities that do not depend on others"
    single_cons, dependent_cons = technology_values()

    "Obtain the production share defined in User"""
    single_perc, dependent_perc = production_share()

    "Obtain the mass balance values of the dependent commodities in Database"
    dep_values = []
    for sust in User.get('Production_Share'):
        if sust in Database.get('Mass_Balance'):
            metric_val = Database['Mass_Balance'][sust]
            temp_list = []
            for row in metric_val.values():
                ran = User['Random_Value']
                if isinstance(row, list) and len(row) == 2 and ran:
                    random_value = round(random.uniform(row[0], row[1]),4)
                    temp_list.append(random_value)
                else:
                    temp_list.append(mean(row))
            dep_values.append(temp_list)

    "Obtain the names of the commodities to generate the columns"
    single_names = []
    dependent_names = []
    for key, value in User['Production_Share'].items():
        if len(value) == 2:
            single_names.append(key)
        elif len(value) > 2:
            dependent_names.append(key)

    "Obtain the consumption for the dependent commodities"
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

    "Obtain the produced quantity of the single commodities and the energy of the dependent commodities"
    for col in single_names + dependent_names:
        sur[col] = 0.0
        sur[col] = sur[col].astype('float64')
    for index, row in sur.iterrows():
        if row['Prod_Dem_Diff (kW)'] > 0:
            for i, col in enumerate(single_names):
                if col in single_names:
                    sur.at[index, col] = row['Prod_Dem_Diff (kW)'] * single_perc[i] / single_cons[i] if single_cons[i] != 0 else 0
            for i, col in enumerate(dependent_names):
                if col in dependent_names:
                    sur.at[index, col] = row['Prod_Dem_Diff (kW)'] * dependent_perc[i] / result_list[i]
    return(sur)

#Transform commodities produced into electrical energy
def to_electricity():

    surplus = final_production()

    "Obtain the energy density of the commodities generated"
    result_dict = {}
    names = surplus.columns.tolist()
    for names in Database['Commodities_to_Electricity']:
        values = Database['Commodities_to_Electricity'][names]
        for key, val in values.items():
            ran = User['Random_Value']
            efficiency_values = val[0]
            energy_density_values = val[1]
            if len(efficiency_values) == 2 and ran:
                efficiency = random.uniform(*efficiency_values)#-----------------MODIFICAR A DISTRIBUCIÓN NORMAL
            else:
                efficiency = mean(efficiency_values)
            if len(energy_density_values) == 2 and ran:
                energy_density = random.uniform(*energy_density_values)#------MODIFICAR A DISTRIBUCIÓN NORMAL
            else:
                energy_density = mean(energy_density_values)

            result = round(efficiency * energy_density, 4)
        result_dict[names] = result

    "Create columns of commodities to electricity"
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

    "Managing the energy of stored fuels"
    reset_accum = False
    for index, row in surplus.iterrows():
        if index == 0:
            continue
        if row['Prod_Dem_Diff (kW)'] < 0:
            if reset_accum:
                surplus.at[index, 'Accum_Burned_Potential (kW)'] = 0
                surplus.at[index, 'Burned_Diff (kW)'] = surplus.at[index, 'Accum_Burned_Potential (kW)'] + row['Prod_Dem_Diff (kW)']
            else:
                surplus.at[index, 'Accum_Burned_Potential (kW)'] = surplus.at[index - 1, 'Accum_Burned_Potential (kW)'] + row['Prod_Dem_Diff (kW)']
            if surplus.at[index, 'Accum_Burned_Potential (kW)'] < 0:
                surplus.at[index, 'Burned_Diff (kW)'] = surplus.at[index, 'Accum_Burned_Potential (kW)']
                surplus.at[index, 'Accum_Burned_Potential (kW)'] = 0
                reset_accum = True
            else:
                reset_accum = False
        else:
            if reset_accum:
                surplus.at[index, 'Burned_Diff (kW)'] = surplus.at[index, 'Accum_Burned_Potential (kW)'] + row['Prod_Dem_Diff (kW)']
                surplus.at[index, 'Accum_Burned_Potential (kW)'] = row['Total_Potential (kW)']
                reset_accum = False
            else:
                surplus.at[index, 'Accum_Burned_Potential (kW)'] = surplus.at[index - 1, 'Accum_Burned_Potential (kW)'] + row['Total_Potential (kW)']

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
    return(surplus)
    
#Export data to an excel file
def excel():

    sur = final_production()
    surplus = to_electricity()

    sur['Date'] = pd.to_datetime(sur['Date']).dt.date
    surplus['Date'] = pd.to_datetime(sur['Date']).dt.date

    "Unity of electricity selected"
    elec_unit = User['Electricity_Output'].upper()
    if elec_unit == 'GW':
        unit = 1000000
    elif elec_unit == 'MW':
        unit = 1000
    elif elec_unit == 'KW':
        unit = 1
    else:
        sys.exit("Check User.yaml because electricity unit is invalid")

    "Convert units of electricity from columns"
    columns_sur = [col for col in sur.columns if 'kW' in col]
    columns_surplus = [col for col in surplus.columns if 'kW' in col]

    sur[columns_sur] = sur[columns_sur].div(unit)
    surplus[columns_surplus] = surplus[columns_surplus].div(unit)

    "Rename column names according to the corresponding unit"
    sur.rename(columns=lambda x: x.replace('kW', elec_unit), inplace=True)
    surplus.rename(columns=lambda x: x.replace('kW', elec_unit), inplace=True)

    for df in [sur, surplus]:
        for column in df.columns:
            if elec_unit not in column:
                if column != 'H2O' and column != 'Date':
                    df.rename(columns={column: f"{column} (Kg)"}, inplace=True)
                elif column == 'H2O':
                    df.rename(columns={column: f"{column} (Lt)"}, inplace=True)

    "Export xlsx"
    file_names = ['Commodities_Production.xlsx', 'Commodities_to_Electricity.xlsx']
    for i, df in enumerate([sur, surplus]):
        road = os.path.join(current_directory, 'Output', file_names[i])
        df.to_excel(road, index=False)

    print("Simulation completed successfully")

if __name__ == "__main__":
    run = excel()