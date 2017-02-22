# Objective

Create load forecasting model that takes temperature into account.

Model should be able to train and then predict next 24 hours.

# Approach

## Model selection

I decided to try non-time series model with lag variables


# EDA

## Missing values

The temperature data only had 12307 values out of 45505 total rows. I could have back-filled, forward-filled, or used linear interpolation for these missing values. I decided to use linear interpolation because this most accurately reflects how temperature ranges in reality.

There was also 7589 missing values from 
