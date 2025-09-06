# Arbitrage-free-interpolation-of-SSVI-slices 
This python project uses realtime Yahoo Finance data for a given ticker to calculate implied volatility and then calibrates an eSSVI model based on this data - using Plotly and Streamlit to display the results. 

## Live Demo on Streamlit
[Access to the application on Streamlit](https://arbitragefreessvi.streamlit.app/)

![eSSVI SPY Surface](SPY_Surface)

## Design
- **Data Fetching**: Uses Yahoo Finance (yfinance) to pull real-time option chain data
- **IV Calculation**: Computes implied volatilities using Brent’s root-finding method
- **Forward Estimation**: Estimates forward prices via robust regression (HuberRegressor) to enforce put–call parity
- **Model Calibration**: Calibrates each maturity slice to the eSSVI model using constrained optimization (SciPy), based on [Zeliade Systems](https://www.zeliade.com/wp-content/uploads/whitepapers/zwp-008-RobustNoArbSSVI.pdf)
- **Parameter Interpolation**: Interpolates parameters across maturities to generate a smooth, arbitrage-free surface
- **Visualization**: Creates interactive 2D and 3D plots with Plotly
- **UI Controls**: Sidebar lets users customize ticker, option type, filters, and display settings

## Skills Developed
- Basic Options and Pricing Theory
- UI Design with Streamlit
- Data Science Techniques and Libraries
- Git Workflow & Repository Management

## License
This project is licensed under the MIT License. You are free to use, modify, and distribute this software as per the conditions stated in the [MIT License](https://opensource.org/licenses/MIT).

---

Created by [Theo Sullivan](https://www.linkedin.com/in/theo-sullivan-4b41ba32a/)
