import streamlit as st
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import models
pd.options.mode.chained_assignment = None

@st.cache(hash_funcs={xr.core.dataset.Dataset: lambda _: None})
def load_grib():
    ds = xr.open_dataset('data/data.grib', engine='cfgrib')
    ds = ds.assign(colog = np.log(ds.co))
    ds['colog'] = ds.colog.assign_attrs(units = 'log(kg kg**-1)')
    ds['colog'] = ds.colog.assign_attrs(long_name = 'CO')
    return ds

ds = load_grib()

@st.cache
def load_input():
    df = pd.read_csv('data/data.csv')
    cols = ['latitude', 'longitude', 't', 'co', 'time']
    df = df[cols]
    return df

input_data = load_input()

@st.cache
def load_train():
    data = pd.read_csv('data/all_merged.csv')
    data['colog'] = np.log(data['co'])
    predictors = ['t', 'colog']
    target = ['concentration']
    X = data[predictors]
    y = data[target]
    y['concentration'] = np.log(y['concentration'])
    return X, y

X, y = load_train()

st.title('Global Aerosol Modeling')


st.header('Background')
st.write('Aerosols have been measured to be responsible for a cooling effect in our climate. \
          A proposed feedback mechanism on how this works is that the biosphere reacts to warming climate by emitting more organic compounds, \
          which leads to enhanced growth of aerosols and to an increase in the cloud droplet concentrations and thus a decrese in net radiative forcing. \
          Number concentrations of particles with dry diameters larger than 100 nm are used as a proxy of cloud condensation nuclei (CCN) number concentrations.')

st.image('images/aerosol_schema.png', use_column_width=True, format='PNG',
         caption='Schematic of the proposed climate feedback mechanism [1]')

st.write('However, the magnitude of this effect remains very unclear as we can see from errors in the measurements that were published in the \
         the Intergovernmental Panel on Climate Change (IPCC) Summary for Policymakers report:')

st.image('images/ipcc.png', use_column_width=True, format='PNG',
          caption='Radiative forcing components [2] where we are interested in the uncertain impact from aerosols')

st.header('Hypothesis')
st.write('Now what if we graph aerosol concentration against temperature? We see a relationship that could be explained by \
          a baseline aerosol concentration from anthropogenic emissions in addition to biogenic growth. \
          This is a natural assumption from the graph below where each differently colored line represent a station for measuring aerosol concentration \
          somewhere in the world. Note that mixed boundary layer burden is displayed instead of aerosol concentration, \
          this is because the height of the boundary layer, within which the aerosols are efficiently mixed, is also a strong function of temperature. \
          The burden comes from multiplying measured concentration with the boundary layer height. \
          A trend can be seen where the baseline aerosol concentration is higher for cities that spew out more emissions than \
          some rural station in the middle of the forest for example. \
          The number of aerosols increase as a function of temperature and could be explained by condensation of vapours  associated with biological activity. \
          ')
st.image('images/b100.png', use_column_width=True, format='PNG', 
         caption='The relationship between air temperature and the burden of CCN-sized aerosol particles, modified from [1]')
st.write('Since aerosol measurents are only conducted at a limited number of stations around the globe the availability of such data is sparse. \
          But using the argument from above we can try to model the aerosol concentration using carbon monoxide (anthropogenic emissions) \
          and temperature (biogenic growth). Here is a proof-of-concept.')

st.header('Input data')
st.dataframe(input_data.style.format({"co": "{:.10f}"}))

times = ('None', '00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00')
time = st.selectbox('Choose a time', times)
time2num = dict(zip(times, (None, 0, 1, 2, 3, 4, 5, 6, 7)))
value = time2num[time]
if value != None:
    fig, (ax1, ax2) = plt.subplots(nrows=2, subplot_kw={'projection': ccrs.Robinson()})
    fig.subplots_adjust(hspace=0.35)

    ds.t.isel(time=value).plot.contourf(ax=ax1, transform=ccrs.PlateCarree())
    ax1.set_global()
    ax1.coastlines()

    ds.colog.isel(time=value).plot.contourf(ax=ax2, transform=ccrs.PlateCarree())
    ax2.set_global()
    ax2.coastlines()

    st.pyplot()

st.header('Predict N100 number concentration')

st.write('Using the following models')
code = '''
sklearn.linear_model.LinearRegression()
sklearn.ensemble.RandomForestRegressor(n_estimators=X_train.shape[1], max_depth=10)
xgboost.XGBRegressor(max_depth=10)
'''
st.code(code, language='python')

st.write('The logarithm is taken on N100 concentration and CO training data to match the relationship in the aerosol number burden graph above. \
          The fitted equation is printed for linear regression.')

regressors = (None, 'Linear Regression', 'Random Forest', 'XGBoost')
regressor = st.selectbox('Choose a model', regressors)

if regressor != None:
    if value == None:
        st.write('No input data, select a time above.')
    else:
        X_train, X_test, y_train, y_test = models.split_data(X, y)
        if regressor == 'Linear Regression':
            model = models.linear_regression(X_train, y_train)
            st.latex(models.linear_equation(model, ['T', '\log CO']))
            r2 = models.score(model, X_test, y_test)
            st.write(f"R2 score: {r2:.2f}")
        elif regressor == 'Random Forest':
            model = models.randomforest(X_train, y_train)
            r2 = models.score(model, X_test, y_test)
            st.write(f"R2 score: {r2:.2f}")
        else:
            model = models.xgboost(X_train, y_train)
            r2 = models.score(model, X_test, y_test)
            st.write(f"R2 score: {r2:.2f}")
        
        # predict on dataset
        df = ds.isel(time=value).to_dataframe()
        df['concentration'] = model.predict(df[['t', 'colog']])
        result = xr.Dataset.from_dataframe(df)
        result['concentration'] = result.concentration.assign_attrs(units = 'log(cm^-3)')
        result['concentration'] = result.concentration.assign_attrs(long_name = 'N100')
        
        # plot result
        fig, ax = plt.subplots(nrows=1, subplot_kw={'projection': ccrs.Robinson()})
        fig.subplots_adjust(hspace=0.35)
        result.concentration.plot.contourf(ax=ax, transform=ccrs.PlateCarree())
        ax.set_global()
        ax.coastlines()

        st.pyplot()

st.header('Image references')

st.write('[1] Paasonen, P., Asmi, A., Petäjä, T. et al. Warming-induced increase in aerosol number concentration likely to moderate climate change. \
          Nature Geosci 6, 438–442 (2013). https://doi.org/10.1038/ngeo1800')

st.write('[2] Group I to the Fourth Assessment Report of the Intergovernmental Panel on Climate Change \
          Solomon, S. ,Qin D. ,Manning M., Chen Z. et. al. IPCC, 2007: Summary for Policymakers. \
          Climate Change 2007: The Physical Science Basis. Cambridge University Press, Cambridge, United Kingdom \
          and New York, NY, USA. https://www.ipcc.ch/report/ar4/wg1')

st.image('images/inarlogo.png', use_column_width=True, format='PNG')
st.image('images/logos.png', use_column_width=True, format='PNG')