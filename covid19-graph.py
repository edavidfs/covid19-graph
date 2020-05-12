import git
import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pycountry
import pathlib
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import ScalarFormatter, LogFormatter, MultipleLocator
from unicodedata import normalize
from datetime import datetime
from time import sleep

CURRENT_PATH = pathlib.Path(__file__).parent.absolute()
print(f'Current path {CURRENT_PATH}')

def data_for_country(country):
    data = pd.DataFrame()
    country_confirmed_df = df_confirmed[df_confirmed['Country/Region'] == country]

    country_confirmed_df = country_confirmed_df.drop(['Province/State', 'Lat', 'Long', 'Country/Region'], axis=1)
    country_confirmed_df = country_confirmed_df.T
    country_confirmed_df.index = pd.to_datetime(country_confirmed_df.index)
    country_confirmed_df = pd.DataFrame(country_confirmed_df.sum(axis=1))
    # display(country_confirmed_df)
    #    display(country_confirmed_df.sum(axis=1))
    country_confirmed_df.columns = [country]
    data['Confirmados'] = country_confirmed_df[country]
    data['Confirmados_por_dia'] = country_confirmed_df[country]-country_confirmed_df[country].shift(1,fill_value=0)
    #display(country_confirmed_df[country].head())
    #display(country_confirmed_df[country].shift(1,fill_value=0).head())
    #display(data['Confirmados_por_dia'].head())
    
    country_recovereds_df = df_recovereds[df_recovereds['Country/Region'] == country]
    country_recovereds_df = country_recovereds_df.drop(['Province/State', 'Lat', 'Long', 'Country/Region'], axis=1)
    country_recovereds_df = country_recovereds_df.T
    country_recovereds_df.index = pd.to_datetime(country_recovereds_df.index)
    country_recovereds_df = pd.DataFrame(country_recovereds_df.sum(axis=1))
    country_recovereds_df.columns = [country]
    data = pd.concat([data, country_recovereds_df[country]], axis=1, sort=False)

    country_deadths_df = df_deadths[df_deadths['Country/Region'] == country]
    country_deadths_df = country_deadths_df.drop(['Province/State', 'Lat', 'Long', 'Country/Region'], axis=1)
    country_deadths_df = country_deadths_df.T
    country_deadths_df.index = pd.to_datetime(country_deadths_df.index)
    country_deadths_df = pd.DataFrame(country_deadths_df.sum(axis=1))
    country_deadths_df.columns = [country]
    data = pd.concat([data, country_deadths_df[country]], axis=1, sort=False)
    data['Muertos_por_dia'] = country_deadths_df[country]-country_deadths_df[country].shift(1, fill_value=0)
    
    data.columns = ['Confirmados', 'Confirmados_por_dia', 'Recuperados', 'Muertos','Muertos_por_dia']
    data = data.fillna(method='pad') 
    data['Activos'] = data['Confirmados'] - data['Recuperados'] - data['Muertos']

    # display(data.head())

    return data

def generate_info_text(country_df):
    confirmados= country_df['Confirmados'].iloc[-1]
    incremento_confirmados = confirmados-country_df['Confirmados'].iloc[-2]
    signo_confirmados = '+' if incremento_confirmados>0 else ''

    
    recuperados = country_df['Recuperados'].iloc[-1]
    incremento_recuperados = recuperados-country_df['Recuperados'].iloc[-2]
    signo_recuperados = '+' if incremento_recuperados>0 else ''

    
    muertes= country_df['Muertos'].iloc[-1]
    incremento_muertes = muertes-country_df['Muertos'].iloc[-2]
    signo_muertes = '+' if incremento_muertes>0 else ''

    
    infectados= country_df['Activos'].iloc[-1]
    incremento_infectados = infectados-country_df['Activos'].iloc[-2]
    signo_infectados = '+' if incremento_infectados>0 else ''
  
    
    fecha = country_df.index[-1]
    
    texto = f'TOTALES (AYER)\n\n'\
            f'Total Confirmados: {confirmados} ({signo_confirmados}{int(incremento_confirmados)}) \n'\
            f'Total Recuperados: {int(recuperados)} ({signo_recuperados}{int(incremento_recuperados)})\n'\
            f'Total Muertes: {int(muertes)} ({signo_muertes}{int(incremento_muertes)})\n'\
            f'Activos Actuales: {int(infectados):>6} ({signo_infectados}{int(incremento_infectados)})\n'\
            f'Actualización: {fecha.strftime("%d %b %Y")}'

    return texto

#url_confirmed = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv'
url_confirmed  = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
url_deadths    = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
url_recovereds = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
url_ccaa = 'https://covid19.isciii.es/resources/serie_historica_acumulados.csv'

df_confirmed = pd.read_csv(url_confirmed)
df_deadths = pd.read_csv(url_deadths)
df_recovereds = pd.read_csv(url_recovereds)
#df_ccaa = pd.read_csv(url_ccaa, skipfooter=1, encoding="iso8859_15")

spain_df = data_for_country('Spain')
italia_df = data_for_country('Italy')
china_df = data_for_country('China')
germany_df = data_for_country('Germany')
francia_df = data_for_country('France')
eeuu_df = data_for_country('US')
uk_df = data_for_country('United Kingdom')
argentina_df = data_for_country('Argentina')
peru_df =  data_for_country('Peru')
mexico_df = data_for_country('Mexico')
colombia_df = data_for_country('Colombia')
cuba_df = data_for_country('Cuba')
chile_df = data_for_country('Chile')
venezuela_df = data_for_country('Venezuela')
bolivia_df = data_for_country('Bolivia')
brasil_df = data_for_country('Brazil')


titles = ['España', 'Italia', 'China', 'Francia', 'Alemania', 'EEUU', 'UK', 'Argentina', 'Peru', 'Mexico', 'Colombia','Cuba', 'Chile', 'Venezuela', 'Bolivia', 'Brasil']
files = ['Espana', 'Italia', 'China', 'Francia', 'Alemania', 'EEUU', 'UK', 'Argentina', 'Peru', 'Mexico', 'Colombia','Cuba', 'Chile','Venezuela', 'Bolivia','Brasil']
data = [spain_df, italia_df, china_df, francia_df, germany_df, eeuu_df, uk_df, argentina_df, peru_df,mexico_df, colombia_df, cuba_df, chile_df, venezuela_df, bolivia_df, brasil_df]

max_y = 990000

pais_id = 0
for country_df in data:
    fig = plt.figure(figsize=(6,10), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    #country_df[['Confirmados', 'Recuperados', 'Muertos', 'Activos']].plot(ax=ax, color=['#1F77B4','#2CA02C','#D62728','#FF7F0E'])
    ax.plot(country_df['Confirmados'], label='Confirmados', color='#1F77B4')
    ax.plot(country_df['Recuperados'], label='Recuperados', color='#2CA02C')
    ax.plot(country_df['Muertos'], label='Fallecidos', color='#D62728')
    ax.plot(country_df['Activos'], label='Activos', color='#FF7F0E')
    
    ax.set_ylabel('Casos Totales')
    ax.set_yscale('log')
    ax.set_ylim(1, max_y)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_major_formatter(ScalarFormatter())

    #set major ticks format
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%d'))
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator())
    xax = ax.get_xaxis()
    xax.set_tick_params(which='major', pad=15)

    
    ax.legend(loc='best')
    ax.grid()    
    ax.text(0.05,0.85, generate_info_text(country_df), transform=ax.transAxes,
            fontsize=9,
            bbox=dict(boxstyle="square",
                      ec='lightsteelblue',
                      facecolor='lightsteelblue',
                      pad= 1
                   ))
    ax2 = ax.twinx()
    ax2.set_yscale('linear')
    ax2.set_ylim(1,2500)
    ax2.set_ylabel('Casos Diarios')
    
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    
    ax2.xaxis.set_minor_formatter(mdates.DateFormatter('%d'))
    ax2.xaxis.set_minor_locator(mdates.WeekdayLocator())
    xax2 = ax2.get_xaxis()
    xax2.set_tick_params(which='major', pad=15)
    
#    #country_df[['Muertos_por_dia']].plot(ax=ax2,linestyle='dashed', color='#D62728')
    ax2.bar(x=country_df.index, height=country_df['Muertos_por_dia'], label='Muertos por dia', color='#D62728', alpha=0.5)
##    country_df.plot.bar(y='Muertos_por_dia',color='#D62728')
    
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fancybox=True, shadow=True, loc='lower left',mode="expand", bbox_to_anchor=(0., 1., 0.999, .102),
             ncol=3)
    
    ax2.legend().remove()
    if (titles[pais_id]=="UK"):
        title = 'Reino Unido'
    else:
        title = titles[pais_id]
            
    ax.set_title(title, pad=45)
    
    filename = titles[pais_id].replace('ñ','n')
    fig.tight_layout()
    fig.savefig(f'{CURRENT_PATH/filename}-test.png')
    pais_id +=1


estados =['Confirmados', 'Recuperados', 'Muertos', 'Activos']
for estado in estados:
    fig = plt.figure(figsize=(6,10), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    
    ax.plot(spain_df[estado], label='España',linewidth=2)
    ax.plot(italia_df[estado], label='Italia',linewidth=1, alpha = 0.5)
    ax.plot(china_df[estado], label='China',linewidth=1, alpha = 0.5)
    ax.plot(eeuu_df[estado], label='EEUU',linewidth=1, alpha = 0.5)
    ax.plot(germany_df[estado], label='Alemania',linewidth=1, alpha = 0.5)
    ax.plot(francia_df[estado], label='Francia',linewidth=1, alpha = 0.5)
    ax.plot(uk_df[estado], label='Reino Unido',linewidth=1, alpha = 0.5)
    ax.plot(argentina_df[estado], label='Argentina',linewidth=1, alpha = 0.5)
    
    ax.set_title(f'Comparativa {estado}')
    ax.set_yscale('log')
    ax.set_ylim(1, max_y)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_major_formatter(ScalarFormatter())

    #set major ticks format
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%d'))
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator())
    xax = ax.get_xaxis()
    xax.set_tick_params(which='major', pad=15)
    
    
    ax.legend(loc='best')
    ax.grid()    
    
    fig.savefig(f'{CURRENT_PATH}/comparativa_{str(estado)}.png')


## ANIMACION CONFIRMADOS


fig = plt.figure(figsize=(6,10), dpi=100)
ax = plt.axes(xlim=(spain_df['Confirmados'].index[0], spain_df['Confirmados'].index[-1]), ylim=(1, max_y))

#ax.plot(spain_df[estado], label='España',linewidth=2)
#ax.plot(italia_df[estado], label='Italia',linewidth=1, alpha = 0.5)
#ax.plot(china_df[estado], label='China',linewidth=1, alpha = 0.5)
#ax.plot(eeuu_df[estado], label='EEUU',linewidth=1, alpha = 0.5)
#ax.plot(germany_df[estado], label='Alemania',linewidth=1, alpha = 0.5)
#ax.plot(francia_df[estado], label='Francia',linewidth=1, alpha = 0.5)
#ax.plot(uk_df[estado], label='Reino Unido',linewidth=1, alpha = 0.5)

espana_plot, = ax.plot([], [], lw=2, label='España')
china_plot, = ax.plot([], [], lw=2, label='China')
italia_plot, = ax.plot([], [], lw=2, label='Italia')
eeuu_plot, = ax.plot([], [], lw=2, label='EEUU')

#ax.set_title(f'Comparativa')
ax.set_yscale('log')
#ax.set_ylim(1, max_y)
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.xaxis.set_major_formatter(ScalarFormatter())

#set major ticks format
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax.xaxis.set_major_locator(mdates.MonthLocator())

ax.xaxis.set_minor_formatter(mdates.DateFormatter('%d'))
ax.xaxis.set_minor_locator(mdates.WeekdayLocator())


xax = ax.get_xaxis()
xax.set_tick_params(which='major', pad=15)

ax.legend(loc='upper left')
ax.grid()    

x = np.array(spain_df.index)
espana_y = np.array(spain_df['Confirmados'])
china_y = np.array(china_df['Confirmados'])
italia_y = np.array(italia_df['Confirmados'])
eeuu_y = np.array(eeuu_df['Confirmados'])

xdata, espana_ydata, china_ydata, italia_ydata, eeuu_ydata = [], [] , [], [], []
ax.set_title(f"Evolución de Confirmados")

def init():
    espana_plot.set_data([], [])
    china_plot.set_data([], [])
    italia_plot.set_data([], [])
    eeuu_plot.set_data([], [])

    
    return espana_plot,china_plot,italia_plot, eeuu_plot

def update(i):
    #print(f'timestep {i}')

    xdata.append(x[i]) 
    espana_ydata.append(espana_y[i]) 
    china_ydata.append(china_y[i]) 
    italia_ydata.append(italia_y[i])
    eeuu_ydata.append(eeuu_y[i])
    
    espana_plot.set_data(xdata, espana_ydata)
    china_plot.set_data(xdata, china_ydata)
    italia_plot.set_data(xdata, italia_ydata)
    eeuu_plot.set_data(xdata, eeuu_ydata)

    return espana_plot, china_plot, italia_plot, eeuu_plot

HOLD_COUNT = 5000
def frame_generator():
    for frame in np.arange(0, spain_df.count()['Confirmados']):
        # Yield the frame first
        yield frame
        # If we should "sleep" here, yield None HOLD_COUNT times
        if frame == spain_df.count()['Confirmados']-1:
            for _ in range(HOLD_COUNT):
                yield frame

try:
    anim = FuncAnimation(fig, update, init_func=init, frames=frame_generator, interval=150)
    anim.save(f"{CURRENT_PATH}/Evolucion_confirmados.gif", writer = 'imagemagick')
except:
    print("ERROR Generando evolucion de Confirmados")
      
## Animacion muertes

fig = plt.figure(figsize=(6,10), dpi=100)
ax = plt.axes(xlim=(spain_df['Muertos'].index[0], spain_df['Muertos'].index[-1]), ylim=(1, max_y))

#ax.plot(spain_df[estado], label='España',linewidth=2)
#ax.plot(italia_df[estado], label='Italia',linewidth=1, alpha = 0.5)
#ax.plot(china_df[estado], label='China',linewidth=1, alpha = 0.5)
#ax.plot(eeuu_df[estado], label='EEUU',linewidth=1, alpha = 0.5)
#ax.plot(germany_df[estado], label='Alemania',linewidth=1, alpha = 0.5)
#ax.plot(francia_df[estado], label='Francia',linewidth=1, alpha = 0.5)
#ax.plot(uk_df[estado], label='Reino Unido',linewidth=1, alpha = 0.5)

espana_plot, = ax.plot([], [], lw=2, label='España')
china_plot, = ax.plot([], [], lw=2, label='China')
italia_plot, = ax.plot([], [], lw=2, label='Italia')
eeuu_plot, = ax.plot([], [], lw=2, label='EEUU')

#ax.set_title(f'Comparativa')
ax.set_yscale('log')
#ax.set_ylim(1, max_y)
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.xaxis.set_major_formatter(ScalarFormatter())

#set major ticks format
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax.xaxis.set_major_locator(mdates.MonthLocator())

ax.xaxis.set_minor_formatter(mdates.DateFormatter('%d'))
ax.xaxis.set_minor_locator(mdates.WeekdayLocator())


xax = ax.get_xaxis()
xax.set_tick_params(which='major', pad=15)

ax.legend(loc='best')
ax.grid()    

x = np.array(spain_df.index)
espana_y = np.array(spain_df['Muertos'])
china_y = np.array(china_df['Muertos'])
italia_y = np.array(italia_df['Muertos'])
eeuu_y = np.array(eeuu_df['Muertos'])

xdata, espana_ydata, china_ydata, italia_ydata, eeuu_ydata = [], [] , [], [], []
ax.set_title(f"Evolución de Fallecidos")

def init():
    espana_plot.set_data([], [])
    china_plot.set_data([], [])
    italia_plot.set_data([], [])
    eeuu_plot.set_data([], [])
    return espana_plot,china_plot,italia_plot, eeuu_plot

def update(i):
    #print(f'timestep {i}')
    xdata.append(x[i]) 
    espana_ydata.append(espana_y[i]) 
    china_ydata.append(china_y[i]) 
    italia_ydata.append(italia_y[i])
    eeuu_ydata.append(eeuu_y[i])
    
    espana_plot.set_data(xdata, espana_ydata)
    china_plot.set_data(xdata, china_ydata)
    italia_plot.set_data(xdata, italia_ydata)
    eeuu_plot.set_data(xdata, eeuu_ydata)
    
    
    return espana_plot, china_plot, italia_plot, eeuu_plot

def frame_generator():
    for frame in np.arange(0, spain_df.count()['Confirmados']):
        # Yield the frame first
        yield frame
        # If we should "sleep" here, yield None HOLD_COUNT times
        if frame == spain_df.count()['Confirmados']-1:
            for _ in range(HOLD_COUNT):
                yield frame
try:
    anim = FuncAnimation(fig, update, init_func=init, frames=frame_generator, interval=150)
    anim.save(f"{CURRENT_PATH}/Evolucion_fallecidos.gif", writer = 'imagemagick')
except:
    print("ERROR Generando evolucion de fallecidos")


## Actualizacion Repositorio

repo = git.Repo(f"{CURRENT_PATH}/")
print(repo.is_dirty())  # check the dirty state
repo.untracked_files
#print(len(repo.remotes))
#print(repo.remotes)
#print(repo.index)
#print(repo)
#print(repo.index.diff(None)) # diff with the working copy

files = repo.git.diff(None, name_only=True)
if files:
    for f in files.split('\n'):
        print(f)
        repo.git.add(f)
if len(files)>1:
    commit_message = f'Actualizacion: {datetime.today()}'
    print(commit_message)
    try:
        repo.git.commit('-m', commit_message)
        print(repo.heads.master)
        print(repo.remotes.origin.url)
#        print(subprocess.run(['pwd'], capture_output=True))
#        print(subprocess.run(['cd', CURRENT_PATH,'/'], capture_output=True))
#        print(subprocess.run(['pwd'], capture_output=True))
#        print(subprocess.run(['ls'], capture_output=True))
        print(subprocess.run(["git","-C",CURRENT_PATH, "push", 'origin', 'master'], capture_output=True))

    #    repo.remotes.origin.push(f'{repo.heads.master}')
    #    ssh_cmd = 'ssh -i id_rsa'
    #    with repo.git.custom_environment(GIT_SSH_COMMAND=ssh_cmd):
    #        repo.remotes.origin.push(f'{repo.heads.master}')

    except ValueError:
        print(f'Error al subir los archivos al repositorio: {ValueError}')

    print("Finish push data")