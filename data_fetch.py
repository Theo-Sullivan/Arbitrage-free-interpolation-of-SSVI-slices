# This file imports data from yahoo finance
# It then validates initial data and creates option chain
# Then finds IV, forward price, bidIV etc and creates new dataframe for processed data




# IMPORTS

import yfinance as yf
import pandas as pd
import datetime 
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from sklearn.linear_model import HuberRegressor




# HELPER FUNCTIONS

# Black-Scholes Equation
def black_scholes(S, K, T, r, sigma, q=0, otype="call"):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if otype == "call":
        call = S * np.exp(-q*T) * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
        return call
    else:
        put = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q*T) * norm.cdf(-d1)
        return put
    

# For finding IV
def solve_for_iv_brent(S, K, T, r, price, otype, q=0, verbose=False, boundaries = True):
    minIV = 1e-5
    maxIV = 5

    # Checking for initial data issues
    if boundaries:
        if T <= 0:
            if verbose:
                    print(f"outlier removed due to T <= 0")
            return np.nan
        

        elif otype == "call":
            if not max(0, S*np.exp(-q*T) - K*np.exp(-r*T)) <= price <= S*np.exp(-q*T):
                if verbose:
                    print(f"outlier removed due to false arbitrage bounds")
                return np.nan

        elif otype == "put":
            if not max(0, K*np.exp(-r*T) - S*np.exp(-q*T))<= price <= K*np.exp(-r*T):
                if verbose:
                    print(f"outlier removed due to false arbitrage bounds")
                return np.nan
              
    # Define objective: option price difference
    def f(sigma):
        return black_scholes(S, K, T, r, sigma, q, otype) - price

    try:
        IV = brentq(f, minIV, maxIV)   # search between 0.00001 and 500% vol
    except ValueError:
        if verbose:
                print(f"outlier removed due to approximation failure")
        return np.nan
    
    # Checking for issues with IV
    if boundaries:
        if IV > maxIV:
            if verbose:
                print(f"outlier removed due to IV > {maxIV}")
            return np.nan

        elif IV < minIV:
            if verbose:
                print(f"outlier removed due to IV < {minIV}")
            return np.nan
    

    return IV


# Fetch the three month risk free rate as a default
def three_month_rate(default=0.041):
    try:
        data = yf.download("^IRX", period="5d", progress=False)
        return data["Close"].iloc[-1].item() / 100   # convert % to decimal
    except:
        return default
    

# Finding forward rates using linear regression put call parity
def linear_regression_F(mergedOptchain, S, verbose=False):
    t_val_to_forward = {}
    t_val_to_r = {}

    risk_free_rate = three_month_rate()

    for t_val, block in mergedOptchain.groupby("T"):
        midPriceCalls = (block['bid_x'] + block['ask_x']) / 2
        midPricePuts  = (block['bid_y'] + block['ask_y']) / 2
        Y = (midPriceCalls - midPricePuts).to_numpy()
        X = block['strike'].to_numpy().reshape(-1, 1)

        X_scaled = X/S
        Y_scaled = Y/S

        # Robust regression
        model = HuberRegressor(max_iter=500).fit(X_scaled, Y_scaled)
        slope = model.coef_[0]
        intercept = model.intercept_ * S

        D = -slope
        F = intercept/(D)

        if not 0 < D <= 1: # This is of note
            r = risk_free_rate
            F = S * np.exp(r*t_val)
        else:
            r = -np.log(D) / t_val

        key = round(t_val,5)
        t_val_to_forward[key] = F
        t_val_to_r[key] = r

    return t_val_to_forward, t_val_to_r


#FETCHING DATA // MAIN FUNCTIONS


# Getting initial option chain for either call or put
def optchain_get_either(optType_: str, tickr_: str, verbose = False, mRange = (0.6,1.4), tRange = (0.1,2)):
    tickr = yf.Ticker(tickr_)
    S = yf.Ticker(tickr_).info['regularMarketPrice']

    ma,mb = mRange
    ta,tb = tRange

    # Fetching Options Error Checking
    try:
        expiryDates = tickr.options
    except Exception as e:
        raise Exception(f"Error fetching options for {tickr_}: {e}")

    if expiryDates == ():
        raise Exception(f"There is no option data for {tickr_} ")

    # CREATING OPTION CHAIN
    opt_chain0 = pd.DataFrame()
    fail = 0


    for expiry in expiryDates:
        
        # Checking if there is option data for a date
        try:
            chain = tickr.option_chain(expiry)
        except Exception as e:
            fail += 1
            if verbose:
                print(f"{fail}. Skippinge expiry {expiry} due to error: {e}")
                continue

        if optType_ == 'call':
            opt_chain = chain.calls
        elif optType_ == 'put':
            opt_chain = chain.puts
        else:        
            raise Exception("Must have option type 'call' or 'put'")

        # Calculating T and adding to opt_chain
        expiryDT = datetime.datetime.strptime(expiry, "%Y-%m-%d").date()
        date = (datetime.datetime.now().date())
        T = (expiryDT - date).days/365
        opt_chain['T'] = T

        #Concatting
        opt_chain0 = pd.concat([opt_chain0, opt_chain])

    # Validating data
    if (opt_chain0['bid'].fillna(0) == 0).all():
        raise Exception("No options are being traded")
    
    opt_chain0 = opt_chain0[(opt_chain0["strike"].between(ma * S, mb * S)) & (opt_chain0["T"].between(ta, tb))]

    safe_block = opt_chain0[
    opt_chain0[['bid', 'ask']].notna().all(axis=1) &   # no NaNs
    (opt_chain0[['bid', 'ask']] != 0).all(axis=1)     # no zeros
    ].copy()

    if safe_block.empty:
        raise Exception("No valid option data after cleaning (all bids or asks are NaN/0)")

    return safe_block







# combine option chains for put and call
def optchainData(optType_, tickr_, verbose, mRange, tRange):
    calls = optchain_get_either('call', tickr_, verbose, mRange, tRange)
    puts = optchain_get_either('put', tickr_, verbose, mRange, tRange)
    
    mergedOptchain = pd.merge(calls, puts, on=["strike", "T"], how="inner") 
    common_keys = mergedOptchain[["strike", "T"]]

    if optType_ == "call":
        opt_chain = calls.merge(common_keys, on=["strike", "T"], how="inner")
    else:
        opt_chain = puts.merge(common_keys, on=["strike", "T"], how="inner")

    return opt_chain, mergedOptchain





# Validating data and creating new dataframe
def impliedVolSurfaceData_eSSVI(optType_, mergedOptChain, tickr_, opt_chain, verbose=False, plot_bidask=False, volume_filter = False, implied_yield = False):
    ivs = []
    logmoneyness = []
    dtes = []
    ws = []
    bid_IVs = []
    ask_IVs = []
    
    bid_IV = np.nan
    ask_IV = np.nan

    # GETTING DATA
    S = yf.Ticker(tickr_).info['regularMarketPrice']
    risk_free_rate = three_month_rate()

    if implied_yield:
        t_val_to_forward, t_val_to_r = linear_regression_F(mergedOptChain, S, verbose = verbose)

    for idx, row in opt_chain.iterrows():        
        K = row["strike"]
        T = row["T"]

        if implied_yield:
            key = round(T,5)
            F = t_val_to_forward.get(key)
            r = t_val_to_r.get(key)
        else:
            r = risk_free_rate
            F = S * np.exp(T * r)
        
        # dict validation
        if pd.isna(F) or pd.isna(r):
            r = risk_free_rate
            F = S * np.exp(r*T)
        
        bid = row['bid']
        ask = row['ask']
        midPrice = 0.5 * (bid + ask)
        M = np.log(K/F)
        
        
        # Bid-Ask Validation since bad data
        if (bid < 0.1 or ask < 0.1) and volume_filter: 
            if verbose:
                print("Removed data point due to low bid, ask or liquidity")
            continue

        
        # Calculating IV
        IV = solve_for_iv_brent(S, K, T, r, midPrice, optType_, verbose=verbose)

        if np.isnan(IV):
            continue

        if plot_bidask:
            bid_IV = solve_for_iv_brent(S, K, T, r, bid, optType_, verbose=verbose, boundaries = False)
            ask_IV = solve_for_iv_brent(S, K, T, r, ask, optType_, verbose=verbose, boundaries = False)

            if np.isnan(bid_IV) or np.isnan(ask_IV):
                ask_IV = np.nan
                bid_IV = np.nan
            
        
        ivs.append(IV)
        logmoneyness.append(M)
        dtes.append(T)
        ws.append(T*IV**2)
        bid_IVs.append(bid_IV)
        ask_IVs.append(ask_IV)



    IVT_data = pd.DataFrame({
        'ivs': ivs,
        'logmoneyness': logmoneyness,
        'dtes': dtes,
        'ws' : ws,
        'bid_IV' : bid_IVs,
        'ask_IV' : ask_IVs,
    })


    return IVT_data








