# -*- coding: utf-8 -*-
"""
Created on Wed May 10 15:26:59 2023

@author: DHANA SRIVALLI
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit

# Define constants
EDUCATION_DATA_PATH = "data/education.csv"
EXPENDITURE_DATA_PATH = "data/gov_expenditure.csv"
YEARS = ["1990", "2000", "2012", "2013", "2014", "2015", "2016", "2017", "2018"]
COUNTRY = "India"
LABELS = ["Literacy Rate", "Government Expenditure on Education"]

# Define utility functions
def read_data(filepath):
    data = pd.read_csv(filepath)
    df = pd.DataFrame(data)
    '''
    Load data from a CSV file into a pandas DataFrame.

    Parameters
    ----------
    fname : String
        file name.

    Returns
    -------
   DataFrame
        DataFrame for the given file name

    '''
    dataT = pd.read_csv(filepath, header=None, index_col=0).T
    dfT = pd.DataFrame(dataT)
    dfT = dfT.rename(columns={"Country Name": "Year"})
    return df, dfT

def convert_to_numeric(df, column_list):
    '''
    In this method we will convert the non numbers to numbers
    Parameters
    ----------
    df : DataFrame
        dataframe that need to update.
    columnlist : list
        A list of columns need to be converted.
    Returns
    -------
    df : DataFrame
        Updated DataFrame in which updated columns to numeric type.
       dfT : DataFrame
        DataFrame with countries as columns.
    '''
    df[column_list] = df[column_list].apply(pd.to_numeric, errors="coerce")
    df = df.fillna(0)
    return df

def objective(x, a, b, c):
    '''
    This is the function which returns the quadratic function
    Parameters
    ----------
    x : list
        Dependent variable for creating the equation.
    a : float
         coefficient of x.
    b : float
        coefficient of x2.
    c : float
        constant.
    Returns
    -------
    pandas.core.series.Series
        returns the quadratic function.
    '''
    return a * x + b * x ** 2 + c

def curve_fit_plot(df_lit_rate, df_gov_exp, country):
    '''
    generating the curve fit for a given country
    Parameters
    ----------
    dft : DataFrame
        countries as columns dataframe.
    df1t : DataFrame
        Countries as columns dataframe.
    country : String
        countryname to plot curve fit.
    
    Returns
    -------
    None.
    '''
    # Convert data to numeric
    df_lit_rate = convert_to_numeric(df_lit_rate, country)
    df_gov_exp = convert_to_numeric(df_gov_exp, country)
    
    # Fit curve
    popt, _ = curve_fit(objective, df_lit_rate[country], df_gov_exp[country])
    df_gov_exp[f"Predicted {country}"] = objective(df_lit_rate[country], *popt)

    # Plot data and curve fit
    plt.figure()
    plt.plot(df_lit_rate[country], df_gov_exp[country], color="red")
    plt.plot(df_lit_rate[country], df_gov_exp[f"Predicted {country}"], color="blue")
    cr = np.std(df_gov_exp[country]) / np.sqrt(len(df_gov_exp[country]))
    plt.fill_between(df_lit_rate[country], df_gov_exp[country] - cr, df_gov_exp[country] + cr, color="b", alpha=0.1)
    plt.xlabel(df_lit_rate.columns[0])
    plt.ylabel(df_gov_exp.columns[0])
    plt.legend(["Expected", "Predicted"], loc="upper right")
    plt.title(f"Curve Fit of {country}")
    plt.show()

def kmeans_cluster(df, no_clusters, labels):
    '''
   Generating the k-means clustering for the given data
   Parameters
   ----------
   df_fit : DataFrame
       dataframe with which one uses k-means.
   noclusters : int
       number of clusters.
   label : list
       list of labels.
   Returns
   -------
   None.
   '''
    kmeans = KMeans(n_clusters=no_clusters)
    kmeans.fit(df)
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_

    # Plot data points and centroids
    plt.figure(figsize=(6.0, 6.0))
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=labels, cmap="Accent")
    for ic in range(no_clusters):
        xc, yc = cen[ic, :]
        plt.plot(xc, yc, "dk", markersize=10)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title("K-means Clustering")
    plt.show()
fname1 = "D:/Datasets/API_SE.ADT.LITR.ZS_DS2_en_csv_v2_5358397.csv"
fname2 = "D:/Datasets/API_SE.PRM.ENRR_DS2_en_csv_v2_5363695.csv"

#Importing warnings so that it may ignore warnings
import warnings
warnings.filterwarnings('ignore')

#Invoking the function to get the data
education_df, education_dfT = read_data(fname2)
govexpenditure_df, govexpenditure_dfT = read_data(fname1)

#Taking the required years for plotting into an array
years = ["1990","2000","2012","2013","2014","2015","2016","2017","2018"]

#Invoking convert_to_numeric method to change the data to numeric form
education_df = convert_to_numeric(education_df, years)
govexpenditure_df = convert_to_numeric(govexpenditure_df, years)

#Plot the curve fit line
country = "India"
curve_fit_plot(education_dfT, govexpenditure_dfT, country)

#Plot the k-means clustering
labels = ["Literacy Rate", "Govt. Expenditure on Education"]
df_fit = pd.merge(education_df["2018"], govexpenditure_df["2018"], right_index=True, left_index=True)
kmeans_cluster(df_fit, 4, labels)

# Generate random data for demonstration
np.random.seed(42)
education_df = pd.DataFrame({"2018": np.random.rand(10)})
govexpenditure_df = pd.DataFrame({"2018": np.random.rand(10)})

# Sort the data by the x-axis variable
sorted_education = education_df.sort_values(by="2018")
sorted_expenditure = govexpenditure_df.loc[sorted_education.index]

# Plot the data as a line plot
plt.figure(figsize=(6.0, 6.0))
plt.plot(sorted_education["2018"], sorted_expenditure["2018"])
plt.xlabel("Literacy Rate")
plt.ylabel("Government Expenditure on Education")
plt.title("Education Expenditure vs. Literacy Rate in 2018")
plt.show()