import streamlit as st
import streamlit as st
import pandas as pd
import plotly.express as px 





#Sidebar


page = st.sidebar.selectbox("Select a page", ['Home', 'Data Overview', 'Exploratory Data Analysis','Model Training and Evaluation','Make Predictions','Final Thoughts'])

#Display Pages

if page == "Home":
    st.title("📊 Airline Satisfaction Predictions")
    st.write("This project is an evaluation and data analysis about an airline dataset that provides information about customer satisfaction. In this project, you will see my exploratory data analysis, my model training and evaluation, and the ability to make predictions on my dataset.")
    st.image('./airline.png') 

elif page == "Data Overview":
   ariline = pd.read_csv('./airline_cleaned.csv')



if page == "Data Overview":


    st.title("Data Overview")

    tab1, tab2, tab3 = st.tabs(["Quick Data Overview","Describing the Data", "Shape of the Data"])
    airline = pd.read_csv('./airline_cleaned.csv')
    with tab1:
        st.title("Quick View of the Data ")
        head = pd.DataFrame(airline.head(50))
        st.dataframe(head)
    with tab2:
        st.title("Describing the Data ")
        describe = pd.DataFrame(airline.describe())
        st.dataframe(describe)

    with tab3:  
        st.title("Shape of the Data ")
        shape_airline = pd.DataFrame(airline.shape)
        st.dataframe(shape_airline)
        st.write("This view of the data provides a breadown of the shape of the data.")


elif page == 'Exploratory Data Analysis':
    st.title('Exploratory Data Analysis')


elif page == 'Model Training and Evaluation':
    st.title
    st.write

elif page == 'Make Predictions':
    st.title
    st.write

elif page == 'Final Thoughts':
    st.title
    st.write


if page == 'Exploratory Data Analysis':
    st.subheader("Select the type of visualization you'd like to explore:")
    eda_type = st.multiselect('Visualization Options', ['Histograms', 'Box Plots', 'Scatterplots', 'Box Plots'])
    airline = pd.read_csv ("./airline_cleaned.csv") 
    obj_cols = airline.select_dtypes(include='object').columns.tolist()
    num_cols = airline.select_dtypes(include='number').columns.tolist()

if 'Histograms' in eda_type:
        st.subheader("Histograms - Visualizing the Data")
        st.plotly_chart(px.histogram(airline, x='Age', color='satisfaction'))

if 'Box Plots' in eda_type:
        st.subheader("Box Plots - Visualizing Numerical Distributions")
        st.plotly_chart(px.box(airline, x='Class', y='Age', color='satisfaction'))

if 'Scatterplots' in eda_type:
        st.subheader("Scatterplots - Visualizing Relationships")
        st.plotly_chart(px.scatter(airline, x='Age', y='Flight Distance', color='satisfaction'))

if 'Box Plots' in eda_type:
        st.subheader("Box Plots - Visualizing Numberical Distributions")
        st.plotly_chart(px.box(airline, x='Class', y='Flight Distance', color='satisfaction')) 
