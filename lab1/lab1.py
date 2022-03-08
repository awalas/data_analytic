from statistics import mean
import pandas as pd
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel
from statistics import mean, median
import random

# Exercise 1
# import file to python
path = 'D:\Studia\MAGISTERKA\semestr_1\data_analytics\lab1\Data1.csv'
df = pd.read_csv(path)

# set first column as the index
df = df.rename(columns={'Unnamed: 0': 'date'})
df['date'] = pd.to_datetime(df['date'])
df.set_index('date',inplace=True)

# plot all columns as time series
df.plot(subplots=True, figsize=(12,8))
plt.suptitle('All data- time series',fontsize=16)
plt.show()

# plot histograms
df.plot.hist(bins=200,subplots=True, figsize=(12,8))
plt.suptitle('All data- histograms', fontsize=16)
plt.show()

# plot KDE-s
df.plot.kde(subplots=True, figsize=(12,8))
plt.suptitle('All data- KDE-s', fontsize=16)
plt.show()

# analysis for columns theta_1 - theta_4 in 2018
df_2018 = df.loc[df.index.year == 2018, ['theta_1','theta_2','theta_3','theta_4']]

df_2018.plot(subplots=True, figsize=(12,8))
plt.suptitle('Data from 2018- time series', fontsize=16)
plt.show()

df_2018.plot.hist(bins=150,subplots=True, figsize=(12,8))
plt.suptitle('Data from 2018- histograms', fontsize=16)
plt.show()

df_2018.plot.kde(subplots=True, figsize=(12,8))
plt.suptitle('Data from 2018- KDE-s', fontsize=16)
plt.show()

# Exercise 2
# Create a dataset
F = 5 # number of letters in the first name
L = 7 # number of letters in the last name
N = F+L
y_list = [0]*F + [1]*L
y = random.sample(y_list, len(y_list))
print(y)
data_set = { 'N' : N, 'y' : y}

# Create a model from code provided
file = 'D:\Studia\MAGISTERKA\semestr_1\data_analytics\lab1\\bern_1.stan'
b_model = CmdStanModel(stan_file=file)

# Sample from the model
b_sample = b_model.sample(data_set)

# Extract theta variable & create its histogram
theta = b_sample.stan_variable('theta')
plt.hist(theta, bins=15)

# Get mean, median and 5% and 95% quantiles of theta & mark it on the histogram
summary = b_sample.summary()

theta_mean = mean(theta)
theta_median = median(theta)
theta_5prcnt = summary['5%']['theta']
theta_95prcnt = summary['95%']['theta']

plt.axvline(theta_mean, color = 'm')
plt.axvline(theta_median, color = 'b')
plt.axvline(theta_5prcnt, color = 'g')
plt.axvline(theta_95prcnt, color = 'y')
plt.title('Theta- histogram')
plt.show()