"""Carreau-Yasuda model fitting functions """

from scipy.optimize import curve_fit
import numpy as np
import pandas as pd


def calc_eta_star(df):
    """Calculates complex viscosity
    """
    df['eta_star'] = (np.sqrt(np.power(df['G_p'], 2) + np.power(df['G_pp'], 2))
                      / df['w'])
    return df


def model(w, eta_0, eta_inf, a, n, lam):
    """ Estimates the modulus of complex-viscosity at particular frequency
        using Carreau-Yasuda model.

        Parameters:
        w = Angular frequency
        eta_0 = zero-shear viscosity
        eta_inf = infinite-shear viscosity
        lam = relaxation time
        a = transition parameter
        n = power law index
    """
    eta_star = eta_inf + (eta_0-eta_inf)/((1+(w*lam)**a)**((1-n)/a))
    eta_star = eta_star.rename('eta_star')
    return eta_star


def fit(df, eta_0=None, eta_inf=1E-4, lam=0.5, a=0.5, n=0.5):
    """ Estimates Carreau-Yasuda model fitting parameters for given LVE data.

        Parameters:
        df = DataFrame containing LVE data
        eta_0 = zero-shear viscosity
        eta_inf = infinite-shear viscosity
        lam = relaxation time
        a = transition parameter
        n = power law index
    """

    if 'eta_star' not in df.columns:
        df = calc_eta_star(df)
    if eta_0 is None:
        eta_0 = df.iloc[0]['eta_star']

    popt, *pcov = curve_fit(model, df['w'], df['eta_star'],
                            p0=[eta_0, eta_inf, a, n, lam],
                            bounds=(0, np.inf),
                            loss='soft_l1', f_scale=0.1)
    out_df = pd.concat([df['w'], model(df['w'], *popt)], axis=1)
    return (out_df, popt)
