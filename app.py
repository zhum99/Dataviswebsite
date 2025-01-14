import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

#split window into 2 columns
st.set_page_config(layout = "wide")
graphcol, datacol = st.columns([6,3])

# Initial DataFrame
df = pd.DataFrame(
    [
        {"x": 1, "y": 2},
        {"x": 2, "y": 3},
        {"x": 3, "y": 4},
    ]
)

#invisible data editor version counter
datakey = 0
def getnew_datakey():  
        global datakey
        datakey += 1
        return datakey
    
def getold_datakey():
        global datakey
        return datakey

# Sidebar for file upload
with st.sidebar:
    #Title
    st.title("Upload & Edit Data")
    st.write("Click 'Upload' to load the data into the DataFrame.")
    st.write("Edit the data in the DataFrame below.")
    # State management for the DataFrame
    if "current_df" not in st.session_state:
        st.session_state.current_df = df

    # Placeholder for the DataFrame
    frame = st.empty()

    # Display editable DataFrame
    with frame.container():
        SideFrame = st.data_editor(st.session_state.current_df, num_rows="dynamic", key=getnew_datakey(), use_container_width=True)
        
    # Button to reset the data to df
        if st.button("Reset Data"):
            st.session_state.current_df = df
            frame.empty()
            with frame.container():
                SideFrame = st.data_editor(st.session_state.current_df, num_rows="dynamic", key=getnew_datakey(), use_container_width=True)

    # File uploader button
    UserFileClicker = st.file_uploader("Upload .csv data here:", type=["csv", "txt"])
    
    # Upload button functionality
    if st.button("Upload"):
        # Load new data from uploaded file if provided
        if UserFileClicker:
            new_df = pd.read_csv(UserFileClicker, header=None)
            st.session_state.current_df = new_df  # Update state with new data
        else:
            st.warning("No file uploaded!")

        # Clear and refresh the displayed DataFrame
        frame.empty()
        with frame.container():
            st.data_editor(st.session_state.current_df, num_rows="dynamic", key=getnew_datakey(), use_container_width=True)

with graphcol.container():
    # Render graph title
    st.title("Graph")
    st.write("The graph below shows the data from the DataFrame.")

#Main UI area
with datacol:
    st.title("Data Select")
    cola1, cola2, cola3 = st.columns(3)
    with cola1:
        x_col = st.selectbox("", SideFrame.columns, placeholder="Select x column:")
        #if x column contains NaN values, replace them with 0
        x_raw = SideFrame[x_col]
        x_check = pd.to_numeric(x_raw, errors='coerce')
        if x_check.isnull().values.any():
            x_sanitized = pd.Series(range(0, x_check.shape[0]))
            x_col_name = x_col + " (Found invalid values, column replaced with enumerated list)"
        else:
            x_sanitized = x_raw
            x_col_name = x_col
    with cola2:
        y_col = st.selectbox("", SideFrame.columns, placeholder="Select y column:")
        #Create sanitized version of the y values in case it contains NaN values
        y_raw = SideFrame[y_col]
        y_check = pd.to_numeric(y_raw, errors='coerce')
        if y_check.isnull().values.any():
            y_sanitized = y_check.fillna(0)
            y_col_name = y_col + " (Found invalid values, values replaced with 0)"
        else:
            y_sanitized = y_raw
            y_col_name = y_col
        
    with cola3:
        with graphcol:
            plot = st.empty()
        if st.button("Render Graph"):
            with plot.container():
                # Plot the selected columns
                plt.plot(x_sanitized, y_sanitized)
                plt.xlabel(x_col_name)
                plt.ylabel(y_col_name)
                st.pyplot(plt)
        with datacol:
            st.title("Render Settings")
            colb1, colb2, colb3 = st.columns(3)
            with colb1:
                selectedfit = st.selectbox("Select fit type:", ["Polynomial", "Exponential", "Logarithmic"])
            with colb2:
                if selectedfit == "Polynomial":
                    #Make max degree of polynomial fit equal to the number of data points up to 10
                    dat_max = min(10, x_sanitized.shape[0] - 1)
                    degree = st.slider("Polynomial Degree:", 1, dat_max, 2)
                    pnaught = np.ones(degree+1)
                    #Create polynomial fit for data
                    def plnmod(x,*params):
                        return sum(p * x**i for i , p in enumerate(params ))
                    popt, pcov = curve_fit(lambda x, *params: plnmod(x, *params), x_sanitized, y_sanitized, p0=pnaught)
                    with colb3:
                        if st.button("Render Fit Curve"):
                            yfit= plnmod(x_sanitized,*popt)
                            leftover=y_sanitized-yfit
                            meansqerror=np.mean(leftover**2)
                            avgerror=np.mean(np.abs(leftover))
                            with datacol:
                                st.write(f"Equation: y = {' + '.join([f'{p:.3f}x^{i}' for i, p in enumerate(popt)])}")
                                st.write(f"(Mean squared error : {meansqerror:.3f}")
                                st.write(f"(Average abs error : {avgerror:.3f}")
                            xfit=np.linspace(min(x_sanitized), max(x_sanitized),1000)
                            fig,ax=plt.subplots()
                            ax.scatter(x_sanitized,y_sanitized,label="Your Data", color="blue")
                            ax.plot(xfit,plnmod(xfit,*popt),label="Fitted Curve",color="black")
                            ax.legend()
                            ax.set_xlabel(x_col_name)
                            ax.set_ylabel(y_col_name)
                            ax.set_title(f"Polynomial fit with Degree of {degree}")
                            plot.empty()
                            with plot.container():
                                st.pyplot(fig)
                elif selectedfit == "Exponential":
                    def expmod(x, b, a):
                        return a*np.exp(b*x)
                    popt = np.polyfit(x_sanitized, np.log(y_sanitized), 1)
                    with colb3:
                        if st.button("Render Fit Curve"):
                            yfit= expmod(x_sanitized,*popt)
                            leftover=y_sanitized-yfit
                            meansqerror=np.mean(leftover**2)
                            avgerror=np.mean(np.abs(leftover))
                            with datacol:
                                st.write(f"Equation: y = {popt[1]:.3f} * e^({popt[0]:.3f}x)")
                                st.write(f"(Mean squared error : {meansqerror:.3f}")
                                st.write(f"(Average abs error : {avgerror:.3f}")
                            xfit=np.linspace(min(x_sanitized), max(x_sanitized),1000)
                            fig,ax=plt.subplots()
                            ax.scatter(x_sanitized,y_sanitized,label="Your Data", color="blue")
                            ax.plot(xfit,expmod(xfit,*popt),label="Fitted Curve",color="black")
                            ax.legend()
                            ax.set_xlabel(x_col_name)
                            ax.set_ylabel(y_col_name)
                            ax.set_title("Exponential fit")
                            plot.empty()
                            with plot.container():
                                st.pyplot(fig)
                else:
                    def logmod(x, b, a):
                        return a + b*np.log(x)
                    popt = np.polyfit(np.log(x_sanitized), y_sanitized, 1)
                    with colb3:
                        if st.button("Render Fit Curve"):
                            yfit= logmod(x_sanitized,*popt)
                            leftover=y_sanitized-yfit
                            meansqerror=np.mean(leftover**2)
                            avgerror=np.mean(np.abs(leftover))
                            with datacol:
                                st.write(f"Equation: y = {popt[1]:.3f} + {popt[0]:.3f} * ln(x)")
                                st.write(f"(Mean squared error : {meansqerror:.3f})")
                                st.write(f"(Average abs error : {avgerror:.3f})")
                            xfit=np.linspace(min(x_sanitized), max(x_sanitized),1000)
                            fig,ax=plt.subplots()
                            ax.scatter(x_sanitized,y_sanitized,label="Your Data", color="blue")
                            ax.plot(xfit,logmod(xfit,*popt),label="Fitted Curve",color="black")
                            ax.legend()
                            ax.set_xlabel(x_col_name)
                            ax.set_ylabel(y_col_name)
                            ax.set_title("Logarithmic fit")
                            plot.empty()
                            with plot.container():
                                st.pyplot(fig)