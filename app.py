### IMPORTS ###

# Module importsx
import streamlit as st

# File imports
from data_fetch import optchainData, impliedVolSurfaceData_eSSVI
from calibration import SVI_model_2d_data, interpolation

### UI ELEMENTS ###

st.title("Arbitrage-free interpolation of SSVI slices")

# Heading
st.sidebar.header('Model Parameters')
st.sidebar.write('Adjust the parameters and viewing options for the SSVI model')


# Sidebar
st.sidebar.header('Ticker and Option Type')
tickr_ = st.sidebar.text_input( # Choose Tickr
    'Enter Ticker Symbol',
    value='^SPX',
    max_chars=10
).upper()

optType_ = st.sidebar.selectbox( # Choose Option Type
    'Select Option Type',
    ('Call', 'Put'))

optType_ = optType_.lower()



st.sidebar.header('Surface Visualization Parameters')
y_axis_option = st.sidebar.selectbox( # Choose Y-axis scale
    'Select Y-axis:',
    ('Moneyness', 'Log Moneyness'))

if y_axis_option == 'Log Moneyness': # Turn into True or False
    logplot = True
else:
    logplot = False

st.sidebar.header('2D Visualization Parameters')
plot2D = st.sidebar.checkbox('Plot individual time slices', value=True) #Choose whether to plot 2D
plot_bidask = st.sidebar.checkbox('Plot Bid/Ask', value=False, disabled=not plot2D) #Bid-Ask choice depending of if plot 2D



st.sidebar.header('Calibration Settings')
volume_filter = st.sidebar.checkbox('Use Volume Filter', value=True) #volume 



st.sidebar.header('Strike Price Filter Parameters')
min_m = st.sidebar.number_input(
    'Minimum Moneyness',
    min_value=0.5,
    max_value=0.90,
    value=0.5,
    step=0.01,
    format="%.1f"
)

max_m = st.sidebar.number_input(
    'Maximum Moneyness',
    min_value=1.1,
    max_value=2.0,
    value=2.0,
    step=0.01,
    format="%.1f"
)

if min_m>= max_m:
    st.sidebar.error('Minimum percentage must be less than maximum percentage.')
    st.stop()



#Params for now!
verbose = False
tLimit = 0.1

### CACHING ####
@st.cache_data(ttl=3600)
def optchainData_cached(optType_, tickr_, verbose=False):
    try:
        return optchainData(optType_, tickr_, verbose)
    except Exception as e:
        raise RuntimeError(f"Error fetching option data: {e}") from e


### MAIN ###
try:
    opt_chain, mergedOptchain = optchainData_cached(optType_, tickr_, verbose)
except Exception as e:
    st.error(f"{e}")  # now prints the cached function message cleanly
    st.stop()

with st.status(label='Computing implied volatility...', expanded=True) as status:
    try:
        IVT_data = impliedVolSurfaceData_eSSVI(optType_, mergedOptchain, tickr_, opt_chain, plot_bidask = plot_bidask, verbose = verbose, volume_filter = volume_filter, oldmRange = (min_m, max_m), tLimit = 0.1)

    except Exception as e:
        st.error(f"Error computing IV data: {e}")
        st.stop()


    status.update(label='Calibrating parameters...')
    try:
        if plot2D:  
            plot_data, figs = SVI_model_2d_data(IVT_data, optType_ ,verbose = verbose, plot = plot2D, plot_bidask = plot_bidask, plot_IV = True)
        else:
            plot_data = SVI_model_2d_data(IVT_data, optType_, verbose = verbose, plot = plot2D, plot_bidask = plot_bidask, plot_IV = True)

    except Exception as e:
        st.error(f"Error calibrating SSVI slices: {e}")
        st.stop()

    status.update(label='Interpolating surface...')
    try:
        main_fig = interpolation(tickr_, plot_data, IVT_data, logplot = logplot)

    except Exception as e:
        st.error(f"Error interpolating 3D SSVI: {e}")
        st.stop()

    ### PLOTTING ###
    try:
        if plot2D:
            for fig in figs:
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(plot_data)

        st.plotly_chart(main_fig)

    except:
        st.error("No data to plot")
        st.stop()


# LINKEDIN 
st.write("---")
st.markdown("Theo Sullivan | https://www.linkedin.com/in/theo-sullivan-4b41ba32a/")





