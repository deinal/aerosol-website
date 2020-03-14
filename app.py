import streamlit as st
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image

@st.cache(hash_funcs={xr.core.dataset.Dataset: lambda _: None})
def load_grib():
    ds = xr.open_dataset('data/data.grib', engine='cfgrib')
    return ds

ds = load_grib()

@st.cache(hash_funcs={xr.core.dataset.Dataset: lambda _: None, xr.core.dataarray.DataArray: lambda _: None})
def get_temperature(ds):
    return ds.t

temp = get_temperature(ds)

@st.cache(hash_funcs={xr.core.dataset.Dataset: lambda _: None, xr.core.dataarray.DataArray: lambda _: None})
def get_carbonmoxide(ds):
    colog = xr.ufuncs.log(ds.co)
    colog = colog.assign_attrs(units = 'log(kg kg**-1)')
    colog = colog.assign_attrs(long_name = 'CO')
    return colog

colog = get_carbonmoxide(ds)

@st.cache(hash_funcs={pd.DataFrame: lambda _: None})
def load_data():
    df = pd.read_csv('data/data.csv')
    df.set_index('time', inplace=True)
    return df

df = load_data()

st.title('Aerosol Modeling: proof of concept')

st.header('Input data')
st.write(df)

st.header('Plot input')

times = ['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00']
time = st.selectbox('Choose a time', times)
time2num = dict(zip(times, np.arange(0, 7)))
value = time2num[time]

fig, (ax1, ax2) = plt.subplots(nrows=2, subplot_kw={'projection': ccrs.Robinson()})
fig.subplots_adjust(hspace=0.35)

temp.isel(time=value).plot.contourf(ax=ax1, transform=ccrs.PlateCarree())
ax1.set_global()
ax1.coastlines()

colog.isel(time=value).plot.contourf(ax=ax2, transform=ccrs.PlateCarree())
ax2.set_global()
ax2.coastlines()

st.pyplot()

st.header('Apply model')

## As formula or code, or both...
st.latex(r'''a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} = \sum_{k=0}^{n-1} ar^k = a \left(\frac{1-r^{n}}{1-r}\right)''')


code = '''def model():
    linearRegressor = LinearRegression()
    linearRegressor.fit(X, y)'''
st.code(code, language='python')

btn = st.button("Celebrate!")
if btn:
    st.balloons()

inarlogo = Image.open('inarlogo.png')
st.image(inarlogo, use_column_width=True, format='PNG')
logos = Image.open('logos.png')
st.image(logos, use_column_width=True, format='PNG')