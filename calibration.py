# This file imports data from data_fetch.py
# Then calibrates eSSVI using method outlined in 'Robust calibration and arbitrage-free interpolation of SSVI slices' Zeliade Systems

# IMPORTS

# Imports
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.optimize import minimize


# HELPER FUNCTIONS

def anchor_points(IVT_datat): #Calculating k*, w* = theta* from ATM option log moneyness
    idx_min = IVT_datat['logmoneyness'].abs().idxmin()
    kStar =  IVT_datat.loc[idx_min, 'logmoneyness']
    thetaStar = IVT_datat.loc[idx_min, 'ws']
    return kStar, thetaStar


def theta_eq(rho,psi, kStar, thetaStar): #working all the intermediate variables out 

    theta = thetaStar - rho * psi * kStar

    return theta


def w_model(rho, psi,k, theta): # finding model implied IV
    phi = psi/theta

    w_model = theta/2 * (1 + rho * phi * k + np.sqrt((phi*k + rho)**2 + (1-rho**2)))

    return w_model


def market_model_difference(rho, psi, logmoneyness, ws, theta): # calculating square loss to then minimise
    sum = 0 
    for i in range(len(ws)):
        k = logmoneyness[i]
        w = ws[i]
        
        sum += abs(w_model(rho,psi,k,theta) - w)

    
    return sum


# Helper bound finders
def LB_calendar_eq(rho, rho_prev, psi_prev): # Lower bound for psi to stop calendar arbitrage
    return max( (psi_prev - rho_prev * psi_prev) / (1 - rho), (psi_prev + rho_prev * psi_prev) / (1 + rho) )


def UB_butterfly_eq(rho, thetaStar, kStar): # Upper bound for psi to stop butterfly arbitrage
    UB_butterfly_conservative = 4 / (1 + abs(rho))
    term_inside_sqrt = (4 * rho**2 * kStar**2) / ((1 + abs(rho))**2) + (4 * thetaStar) / (1 + abs(rho))
    
    if term_inside_sqrt < 0:
        print("Value error due to negative sqrt in UB_butterfly calculation")
        return UB_butterfly_conservative

    UB_butterfly_crazy = (-2 * rho * kStar) / (1 + abs(rho)) + np.sqrt(term_inside_sqrt)    

    return min(UB_butterfly_crazy, UB_butterfly_conservative)


def b_theta_eq(rho, kStar, thetaStar): # B_theta value changes based on phi (checking will be done in different function) 
    return thetaStar / (rho * kStar)


# MAIN FUNCTIONS // CALIBRATION

# Calibrates params for a given timeslice
def param_solver(IVT_datat, logmoneyness, ws, rho_prev, psi_prev, verbose = False): 
    kStar, thetaStar = anchor_points(IVT_datat)

    # Defining bounds
    def b_theta(rho):
        return b_theta_eq(rho, kStar, thetaStar)


    # Defining LB_calendar depending on rho_prev, psi_prev
    def LB_calendar(rho):
        return -100
   
    if not np.isnan(rho_prev): # Checking if there are new rho_prev values
        def LB_calendar(rho):
            return LB_calendar_eq(rho, rho_prev, psi_prev)

    def UB_butterfly(rho):
        return UB_butterfly_eq(rho, thetaStar, kStar)


    # Defining vals for minimising function
    def f(v):
        rho, psi = v
        kStar, thetaStar = anchor_points(IVT_datat)
        theta = theta_eq(rho, psi, kStar, thetaStar)

        return market_model_difference(rho, psi, logmoneyness, ws, theta)
    
    def lowerbound(rho):
        if rho * kStar > 0:
            return LB_calendar(rho)
        
        else:
            return max(LB_calendar(rho), b_theta(rho))
        
    def upperbound(rho):
        if rho * kStar > 0:
            return min(UB_butterfly(rho), b_theta(rho))
        
        else:
            return UB_butterfly_eq(rho, thetaStar, kStar)
        
    constraints = [
        {"type": "ineq", "fun": lambda v: v[1] - lowerbound(v[0])},  # y - g1(x) >= 0
        {"type": "ineq", "fun": lambda v: upperbound(v[0]) - v[1]},  # g2(x) - y >= 0
    ]

    bounds = [(-0.99,0.99), (None, None)] 

    # Initial guess same as paper
    x0 = [-0.224, 0.12]

    res = minimize(f, x0, bounds=bounds, constraints=constraints)

    # Returning vals

    if not res.success:
        raise Exception("Failed to find rho value")

    rho, psi = res.x

    difference = res.fun


    return rho, psi, difference



# Using param_solver to calculate params then storing as table 
def SVI_model_2d_data(IVT_data, optType_, plot_IV = True, plot_bidask = False, verbose = False, plot=True):
    # Data for table
    thetas = []
    rhos = []
    psis = []
    t_vals = []
    differences = []
    figs = []

    # Initial value
    fail_count = 0
    rho_prev, psi_prev = np.nan, np.nan


    # Looping over different t_vals
    for t_val, IVT_datat in IVT_data.groupby("dtes"):
        # Data for graph
        logmoneyness_for_graph = IVT_datat['logmoneyness'].to_numpy()
        ws_for_graph = IVT_datat['ws'].to_numpy()
        ivs_for_graph = IVT_datat['ivs'].to_numpy()

        # Data for calibration
        try:
            if optType_ == 'call':
                IVT_datat_calibration = IVT_datat[IVT_datat['logmoneyness'] >= -0.1]
            else:
                IVT_datat_calibration = IVT_datat[IVT_datat['logmoneyness'] <= 0.1]
        except:
            continue

        if len(IVT_datat_calibration) < 5:
            continue

        logmoneyness_for_calibration = IVT_datat_calibration['logmoneyness'].to_numpy()
        ws_for_calibration = IVT_datat_calibration['ws'].to_numpy()


        if plot_bidask:
            bids = IVT_datat['bid_IV'].to_numpy()
            asks = IVT_datat['ask_IV'].to_numpy()


        kStar, thetaStar = anchor_points(IVT_datat_calibration)

        try:
            rho, psi, difference = param_solver(IVT_datat_calibration, logmoneyness_for_calibration, ws_for_calibration, rho_prev, psi_prev, verbose = verbose)
        except:
            continue # We ignore this timeslice 

        rho_prev, psi_prev = rho, psi

        # Checking for param_solver failure
        if np.isnan(rho):
            if verbose:
                fail_count += 1
                print(f"{fail_count} fail")
            continue


        theta_fit = theta_eq(rho, psi, kStar, thetaStar)

        # Data for table
        thetas.append(theta_fit)
        psis.append(psi)
        rhos.append(rho)
        t_vals.append(t_val)
        differences.append(difference)


        if plot:
            # Compute fitted w for all log-moneyness points
            if plot_IV:
                y_fit = [np.sqrt(w_model(rho, psi, k, theta_fit) / t_val) for k in logmoneyness_for_graph]
                y_points = ivs_for_graph

            else:
                y_fit = [w_model(rho, psi, k, theta_fit) for k in logmoneyness_for_graph]
                y_points = ws_for_graph

            # Graphin individual slices
            fig = go.Figure()

            # Plotting bid ask
            if plot_bidask and plot_IV:

                fig.add_trace(go.Scatter(
                    x=logmoneyness_for_graph,
                    y=bids, mode='markers', 
                    name='Bid IV', 
                    marker=dict(color="#e00025", opacity=0.9, size=6, line=dict(width=0.4, color="white"))
                ))
                
                fig.add_trace(go.Scatter(
                    x=logmoneyness_for_graph, y=asks,
                    mode='markers', name='Ask IV',
                    marker=dict(color="#16830c", opacity=0.9, size=6, line=dict(width=0.4, color="white"))
                ))

            else:
                fig.add_trace(go.Scatter(
                    x=logmoneyness_for_graph, y=y_points,
                    mode='markers', name='Market data',
                    marker=dict(color="#e00025", opacity=0.9, size=6, line=dict(width=0.4, color="white"))
                ))
                
            fig.add_trace(go.Scatter(
                x = logmoneyness_for_graph, y = y_fit,
                mode='lines', name='SVI fit',
                line = dict(color="#0059FF", width=2)

            ))

            fig.update_layout(
                title=f'SVI Model Surface T = {t_val:.3f}',
                xaxis_title='Log-moneyness',
                yaxis_title="IV" if plot_IV else "Total implied variance (w)",
                template="plotly"
            )

            figs.append(fig)

    # Data for table
    plot_data = pd.DataFrame({
        'Time to maturity' : t_vals,
        'Market model difference' : differences,
        'Theta' : thetas,
        'Rho' : rhos,
        'Psi' : psis
    })

    if plot:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=t_vals,
            y=thetas,
            mode='lines',
            name='Theta',
            line=dict(color='blue', width=2)
        ))


        fig.add_trace(go.Scatter(
            x=t_vals,
            y=psis,
            mode='lines',
            name='Psi',
            line=dict(color='green', width=2)
        ))

        # Add Rho trace (orange)
        fig.add_trace(go.Scatter(
            x=t_vals,
            y=rhos,
            mode='lines',
            name='Rho',
            line=dict(color='red', width=2)
        ))

        # Update layout
        fig.update_layout(
            title='Evolution of Implied Volatility Parameters (eSSVI)',
            xaxis_title='Time to maturity',
            yaxis_title='Parameter value',
            template='plotly',
            legend=dict(title='Parameters'),
        )
            
        figs.append(fig)

        return plot_data, figs
    
    else:
        return plot_data




# Interpolating parameters over time slices to build 3d plot
def interpolation(tickr_, plot_data, IVT_data, logplot = False):  
    thetas = plot_data['Theta']
    rhos = plot_data['Rho']
    psis = plot_data['Psi']
    t_vals = plot_data['Time to maturity']


    # Functions for parameters
    def theta_eq(x):
        return np.interp(x, t_vals, thetas)
    
    def psi_eq(x):
        return np.interp(x, t_vals, psis)
    
    def rho_eq(x):
        return np.interp(x, t_vals, rhos * psis) / psi_eq(x)
    
    # Finding mins, maxs
    IVT_data = IVT_data[IVT_data["dtes"].isin(t_vals)]

    min_bound_k = IVT_data["logmoneyness"].min()
    max_bound_k = IVT_data["logmoneyness"].max()
    
    
    # Creating meshgrid
    epsilon = 0.01
    min_bound_t = min(t_vals) + epsilon
    max_bound_t = max(t_vals) - epsilon

    min_bound_k = min_bound_k + epsilon
    max_bound_k = max_bound_k - epsilon

    t = np.linspace(min_bound_t, max_bound_t, 1000)
    k = np.linspace(min_bound_k, max_bound_k, 1000)

    T, K = np.meshgrid(t,k)

    #Finding IVs
    W = w_model(rho_eq(T), psi_eq(T), K, theta_eq(T))
    IV = np.sqrt(W / T)

    # If logplot=False, transform x-axis to Forward Price / Strike Price instead of log(Strike Price/ Forward Price) <- Notation is strange 
    if logplot:  
        X_axis = K  
        x_label = 'Log Moneyness (log(Strike / Forward))' 
    else:  
        X_axis = np.exp(K)  
        x_label = 'Moneyness (Strike / Forward)'  

    # Surface
    fig = go.Figure(data=[go.Surface(x=X_axis, y=T, z=IV, colorscale="Viridis", opacity=0.8)])

    # Configuring plotly graph
    fig.update_layout(
        title=f"Implied Arbitrage Free Volatility Surface for {tickr_} Options",
        scene=dict(
            xaxis_title=x_label,
            yaxis_title="Time to Expiration",
            zaxis_title="Implied Volatility",
            xaxis=dict(
                range=[min_bound_k, max_bound_k] if logplot else [np.exp(min_bound_k), np.exp(max_bound_k)]
            ),
            yaxis=dict(range=[min_bound_t, max_bound_t]),
            camera=dict(eye=dict(x=1.2, y=1.2, z=0.8))  # sets the view angle
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        width=900,
        height=700
    )

    # --- Show Plot ---
    return fig















