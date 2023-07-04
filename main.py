import numpy as np
from arch import arch_model
from parametricGarch import Garch

# not required in the function
import pandas as pd
#import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


# Downloading the S&P stock index

# Download S&P 500 stock data
stocks_data = yf.download('^GSPC', start='2012-01-01', end='2022-12-31')

#extract close prices
stocks_data = stocks_data['Close']

# Calculate the returns and drop empty rows
returns = stocks_data.pct_change().dropna() * 100

# Print the downloaded data
print(returns.head())


# Plot of the stocks data and the returns
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].plot(stocks_data)
axs[0].set_title('Stocks Data')

axs[1].plot(returns)
axs[1].set_title('Returns')

plt.show()



# Call Garch function
garch_model = Garch(returns)
print(garch_model)

print(garch_model.forecast_mean)


print(garch_model.standardised_residuals)

garch_model.bootstrap(1000)
print(garch_model.bootstrap_samples)

print(garch_model.estimate_risk(0.95))

