import os
import sys
import warnings
import numpy as np
from prophet import Prophet
from prophet.plot import seasonality_plot_df
from yaml import SafeLoader
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

try: #LOAD CONFIG FILE
    with open(sys.argv[1],encoding="utf-8") as f:
        data_yml = yaml.load(f, Loader=SafeLoader)
        print("Input loaded successfully")
        print(f"Archive: {data_yml['data']}")
except: #IF NO FILE IS PASSED AS ARGUMENT
    print("Input file error")
    exit()

def Decomposition(m,energy): #DECOMPOSITION OF SEASONALITY
    # name = name of the seasonality component you want (yearly,monthly,etc...)
    # m = the model object
    # will return three arrays: days, months, and the seasonality component
    start = pd.to_datetime('2019-01-01 0000')
    name = next(iter(m.seasonalities))
    if name == 'yearly':
        plot_points = 366
        period = m.seasonalities[name]['period']
        end = start + pd.Timedelta(days=period)
    elif name == 'daily':
        plot_points = 365 * 24
        period = m.seasonalities[name]['period']
        end = start + pd.Timedelta(days=period)
    elif name == 'weekly':
        period = m.seasonalities[name]['period']
        end = start + pd.Timedelta(days=period)
        plot_points = 366
    days = pd.to_datetime(np.linspace(start.value, end.value, 366))
    df_y = seasonality_plot_df(m, days)
    seas = m.predict_seasonal_components(df_y)
    save_dict = {}
    save_dict["per"] = seas[name].values[:-1]
    save_dict['day'] = df_y['ds'].dt.day[:-1]
    save_dict['month'] = df_y['ds'].dt.month[:-1]
    if name == 'daily':
        save_dict['time'] = df_y['ds'].dt.hour[:-1]
        return (save_dict['day'], save_dict['month'], save_dict["per"], save_dict['time'])
    else:
        return(save_dict['day'],save_dict['month'],save_dict["per"],0)

def create_pattern_df(df,energy,df_pattern): #CREATE pattern DATAFRAME
    m = Prophet(seasonality_mode='multiplicative')
    df["ds"] = pd.to_datetime(df["ds"],utc=True).dt.tz_localize(None)
    m.fit(df)
    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)
    c, a, b,d = Decomposition(m, energy)  #
    df_pattern["month"] = a
    df_pattern["day"] = c
    if type(d) != int:
        df_pattern["time"] = d
    df_pattern[f"{energy}_per"] = b
    names = pd.DataFrame(df_pattern)
    if data_yml["plot"] == True:
        if not os.path.exists(energy):#CREATE FOLDER
            os.mkdir(energy)#CREATE FOLDER
        m.plot(forecast).savefig(f'{energy}\{energy}_Figure.png')  # plot model
        m.plot_components(forecast).savefig(f"{energy}\{energy}_components.png")  # plot model components
    return names
def fill_df_pattern(df,df_pattern):#FILL pattern DATAFRAME
   
    if os.path.isfile(f'output/{data_yml["pattern"]}'):
        print("Patterns file, already created")
        exit()

    energys = (list(map(lambda x: list(x.keys()), data_yml["energy"]["Predict"])))
    energys2 = (list(map(lambda x: x[0], energys)))  # CAN BE BETTER PLEASE CHANGE
    if not os.path.exists("output"):
        os.mkdir("output")
    writer = pd.ExcelWriter('output/Data_Original.xlsx')
    for energy in energys2:
        if energy != "Nuclear" or energy != "Gas":
            df2 = df[energy]
            df2 = pd.DataFrame({"ds": df[data_yml["date"]], "y": df2})
            try:
                df2["ds"] = pd.to_datetime(df2["ds"],utc=True).dt.tz_localize(None)
            except:
                df2["ds"] = pd.to_datetime(df2["ds"])
            df2.to_excel(writer,sheet_name=energy)
            df_pattern = create_pattern_df(df2,energy,df_pattern)
    df3 = pd.read_csv("data/Demand.csv")
    df3 = df3.rename(columns={"Date": "ds", "demand": "y"})
    df_pattern = pd.concat([df_pattern,create_demand_pattern(df3,writer)],axis=1)
    if not os.path.exists("output"):
        os.mkdir("output")
    writer._save()
    df_pattern.to_csv("output/Pattern.csv", index=False)

def create_demand_pattern(df,writer):
   # df["ds"] = df["ds"].dt.tz_localize(None)
    m = Prophet(seasonality_mode='multiplicative')
    m.fit(df)
    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)
    c, a, b,d= Decomposition(m, "Demand")  #
    df_pattern = {}
    #df_pattern["month"] = a
    #df_pattern["day"] = c
    df_pattern["Demand_per"] = b
    names = pd.DataFrame(df_pattern)
    df.to_excel(writer, sheet_name="Demand")

    if data_yml["plot"] == True:
        if not os.path.exists("Demand"):#CREATE FOLDER
            os.mkdir("Demand")#CREATE FOLDER
        m.plot(forecast).savefig(f'Demand\Demand_Figure.png')  # plot model
        m.plot_components(forecast).savefig(f"Demand\Demand_components.png")  # plot model components
    return names

if __name__ == '__main__':
    with open(sys.argv[1], "r",encoding="utf-8") as ymlfile:
        data_yml = yaml.safe_load(ymlfile)#LOAD CONFIG FILE
        full_df = pd.read_csv(data_yml["data"])#LOAD DATA
        df_pattern = {}
        fill_df_pattern(full_df, df_pattern)#FILL pattern DATAFRAME