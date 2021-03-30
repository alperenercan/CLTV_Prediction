##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################

# 4 Adımda CLTV

# 1. Data Preperation
# 2. BG-NBD Modeli ile Expected Sale Forecasting değerlerini hesapla.
# 3. Calculate the  Expected Average Profit values with Gamma - Gamma model.
# 4. Calculate CLTV for specific time period with BF-NBD and Gamma - Gamma models.

##############################################################
# 1. Data Preperation
##############################################################
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df_= pd.read_excel("D:/MVK/week3/Dosyalar/online_retail_II.xlsx",
                sheet_name="Year 2010-2011")
df = df_.copy()

df = df[df["Country"] == "United Kingdom"]
# Dataset is bigger than I can study so I focused on United Kingdom



def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def crm_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]

    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")

    dataframe["TotalPrice"] = dataframe["Quantity"] * df["Price"]
    return dataframe

df = crm_prep(df)

#############################################
# 2. RFM Table

#############################################

def rfm_table(dataframe):
    """
     !!! nunique() is used for RFM's frequencies
    :param dataframe:
    :return:
    """
    today_date = dt.datetime(2011, 12, 11)

    rfm = dataframe.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                         lambda date: (today_date - date.min()).days],
                                         'Invoice': lambda num: num.nunique(),
                                         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

    rfm.columns = rfm.columns.droplevel(0)
    rfm.head()

    ## recency_cltv_p
    rfm.columns = ['recency_cltv_p', 'T', 'frequency', 'monetary']
    """
        frequency: array_like
            the frequency vector of customers' purchases
            (denoted x in literature).
        recency: array_like
            the recency vector of customers' purchases
            (denoted t_x in literature).
        T: array_like
            customers' age (time units since first purchase)
    """


    rfm["monetary"] = rfm["monetary"] / rfm["frequency"]

    rfm.rename(columns={"monetary": "monetary_avg"}, inplace=True)


    # Calculating of WEEKLY RECENCY and WEEKLY_T for BG/NBD model
    rfm["recency_weekly_p"] = rfm["recency_cltv_p"] / 7
    rfm["T_weekly"] = rfm["T"] / 7

    # CONTROL
    rfm = rfm[rfm["monetary_avg"] > 0]

    # If the  freq is greater than 1, it is our customer.
    rfm = rfm[(rfm['frequency'] > 1)]
    # In BG/NBD dtype of frequency must be integer
    rfm["frequency"] = rfm["frequency"].astype(int)

    return rfm

rfm = rfm_table(df)


##############################################################
# 3. Establishing of  BG/NBD Model
##############################################################
"""
It is about frequency, it doesn't consider the monetary. 
"""
bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(rfm['frequency'],
        rfm['recency_weekly_p'],
        rfm['T_weekly'])


def predict(months):
    if (type(months).__name__!="list"):
        months = [months]
    for month in months:
        rfm["exp_sales_" + str(month) + "_month"] = bgf.predict(month * 4, \
            rfm['frequency'],rfm['recency_weekly_p'],rfm['T_weekly'])

predict([1,3,6,12])

rfm.head(10)

plot_period_transactions(bgf)
plt.show()




##############################################################
# 4. Establishing of GAMMA-GAMMA Model
# It is used for prediction of expected average profit
##############################################################
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(rfm['frequency'], rfm['monetary_avg'])

ggf.conditional_expected_average_profit(rfm['frequency'],
        rfm['monetary_avg']).sort_values(ascending=False).head(10)



def GG_Predictor(months):
    if (type(months).__name__!="list"):
        months = [months]
    cltv = pd.DataFrame()
    for month in months:
        cltv[str(month) + "_months"] = ggf.customer_lifetime_value(bgf,
                                           rfm['frequency'],
                                           rfm['recency_weekly_p'],
                                           rfm['T_weekly'],
                                           rfm['monetary_avg'],
                                           time=month,  # Count of month which we want to predict
                                           freq="W",    # Our Tenor (T) is weekly so I wrote "W"
                                           discount_rate=0.01)
    return cltv

cltv = GG_Predictor([1,3,6,12])


cltv = cltv.reset_index()
cltv.sort_values("6_months", ascending=False).head(10)


rfm_cltv_final = rfm.merge(cltv, on="Customer ID", how="left")
rfm_cltv_final.head()

rfm_cltv_final.sort_values("6_months", ascending=False).head(10)
rfm_cltv_final.sort_values("6_months", ascending=False).tail(10)


rfm_cltv_final["segment_for_6"] = pd.qcut(rfm_cltv_final["6_months"],3, labels=["C", "B", "A"])
