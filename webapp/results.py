import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
pd.options.mode.chained_assignment = None
import models

def load_grib():
    ds = xr.open_dataset('data/data.grib', engine='cfgrib')
    ds = ds.assign(colog = np.log(ds.co))
    ds['colog'] = ds.colog.assign_attrs(units = 'log(kg kg^-1)')
    ds['colog'] = ds.colog.assign_attrs(long_name = 'CO')
    return ds

ds = load_grib()

for value in (0, 1, 2, 3, 4, 5, 6, 7):
        
    fig, (ax1, ax2) = plt.subplots(nrows=2, subplot_kw={'projection': ccrs.Robinson()})
    fig.subplots_adjust(hspace=0.35)

    ds.t.isel(time=value).plot.contourf(ax=ax1, transform=ccrs.PlateCarree())
    ax1.set_global()
    ax1.coastlines()

    ds.colog.isel(time=value).plot.contourf(ax=ax2, transform=ccrs.PlateCarree())
    ax2.set_global()
    ax2.coastlines()

    plt.savefig(f'images/input/{value}.png')
    plt.close(fig)


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
X_train, X_test, y_train, y_test = models.split_data(X, y)

for regressor in (models.linear_regression, models.randomforest, models.xgboost):
    model = regressor(X_train, y_train)
    r2 = models.score(model, X_test, y_test)
    name = regressor.__name__
    with open(f'images/{name}/info.txt', 'w') as info:
        if name == 'linear_regression':
            equation = models.linear_equation(model, ['T', '\log CO'])
            info.write(f"{equation}\nR2 score: {r2:.3f}")
        else:
            info.write(f"R2 score: {r2:.3f}")

    for value in (0, 1, 2, 3, 4, 5, 6, 7):
        df = ds.isel(time=value).to_dataframe()
        df['concentration'] = model.predict(df[['t', 'colog']])
        result = xr.Dataset.from_dataframe(df)
        result['concentration'] = result.concentration.assign_attrs(units = 'log(cm^-3)')
        result['concentration'] = result.concentration.assign_attrs(long_name = 'N100')
        
        fig, ax = plt.subplots(nrows=1, subplot_kw={'projection': ccrs.Robinson()})
        fig.subplots_adjust(hspace=0.35)
        result.concentration.plot.contourf(ax=ax, transform=ccrs.PlateCarree())
        ax.set_global()
        ax.coastlines()

        plt.savefig(f'images/{name}/{value}.png')
        plt.close(fig)
    



