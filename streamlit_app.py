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
        st.subheader("Box Plots - Visualizing Numberical Distributions II")
        st.plotly_chart(px.box(airline, x='Class', y='Flight Distance', color='satisfaction')) 


elif page == 'Model Training and Evaluation':
    st.title('Model Training and Evaluation')
elif page == 'Make Predictions':
    st.title('Make Predictions')

elif page == 'Final Thoughts':
    st.title('Final Thoughts')

    

if page == 'Model Training and Evaluation':
        st.subheader("🛠️ Model Training and Evaluation")
        st.write("This section will cover various training models for the airline data such as:K-Nearest Neighbors, Logistic Regression, Random Forest. ")
        st.image('./airline3.jpg') 

   
        # Import necessary libraries
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import ConfusionMatrixDisplay
        from sklearn.preprocessing import StandardScaler

        st.subheader("🛠️ Model Training and Evaluation")

        # Sidebar for model selection
        st.sidebar.subheader("Choose a Machine Learning Model")
        model_option = st.sidebar.selectbox("Select a model", ["K-Nearest Neighbors", "Logistic Regression", "Random Forest"])

        # Prepare the data
        airline = pd.read_csv('./airline_cleaned.csv')
        X=airline.drop('satisfaction', axis=1)
        y=airline['satisfaction']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(airline.drop('satisfaction', axis=1), airline['satisfaction'], test_size=0.2, random_state=42)

        # Scale the data
        # Identify categorical and numerical features
        categorical_features = X_train.select_dtypes(include=['object']).columns
        numerical_features = X_train.select_dtypes(exclude=['object']).columns

        # Create transformers for numerical and categorical features
        numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore')) # sparse=False for compatibility with KNN
        ])

        # Combine transformers using ColumnTransformer
        preprocessor = ColumnTransformer(
        transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
        ])
        # Create a pipeline with preprocessing and KNN
        pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', KNeighborsClassifier())
        ])
        # Initialize the selected model
        if model_option == "K-Nearest Neighbors":
            k = st.sidebar.slider("Select the number of neighbors (k)", min_value=1, max_value=20, value ='')
            model = KNeighborsClassifier(n_neighbors=k)
        elif model_option == "Logistic Regression":
            model = LogisticRegression()
            k = st.sidebar.slider("Select the number of neighbors (k)", min_value=1, max_value=20, value ='')
            model = KNeighborsClassifier(n_neighbors=k)
        elif model_option == "Random Forest":
            model = RandomForestClassifier()
            k = st.sidebar.slider("Select the number of neighbors (k)", min_value=1, max_value=20, value ='')
            model = KNeighborsClassifier(n_neighbors=k)


            # Fit the pipeline to the training data
            from sklearn.pipeline import Pipeline
       
        
            pipeline.fit(X_train, y_train)
            pipeline.fit(X_test,y_test)

            # Display training and test accuracy
            st.write(f"**Model Selected: {model_option}**")
            st.write(f"Training Accuracy: {pipeline.score(X_train, y_train):.2f}")
            st.write(f"Test Accuracy: {pipeline.score(X_test, y_test):.2f}")

         # Display confusion matrix
            import matplotlib.pyplot as plt
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            y_pred = pipeline.predict(X_test)
            ConfusionMatrixDisplay.from_predictions( y_test, y_pred, ax=ax, cmap='Blues')
            st.pyplot(fig)

        if model_option == "Logistic Regression":
            #fit

            pipeline_lr = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression())
            ])

            pipeline_lr.fit(X_train, y_train)

         # Display training and test accuracy
            st.write(f"**Model Selected: {model_option}**")
            st.write(f"Training Accuracy: {pipeline_lr.score(X_train, y_train):.2f}")
            st.write(f"Test Accuracy: {pipeline_lr.score(X_test, y_test):.2f}")

            # Display confusion matrix
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            y_pred = pipeline_lr.predict(X_test)
            ConfusionMatrixDisplay.from_predictions( y_test, y_pred, ax=ax, cmap='Reds')
            st.pyplot(fig)

            if model_option == "Random Forest":
                # Identify categorical and numerical features
                categorical_features = X_train.select_dtypes(include=['object']).columns
                numerical_features = X_train.select_dtypes(exclude=['object']).columns

                # Create transformers for numerical and categorical features
                numerical_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
                ])

                categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))  
                ])

                # Combine transformers using ColumnTransformer
                preprocessor = ColumnTransformer(
                transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
                ])

                # Create a pipeline with preprocessing and RandomForestClassifier
                pipeline_rf = Pipeline(steps=[  # Changed pipeline name to pipeline_rf
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier()) 

                ])
                # Fit the pipeline to the training data
                pipeline_rf.fit(X_train, y_train)  # Using pipeline_rf for fitting

                # Display training and test accuracy
                st.write(f"**Model Selected: {model_option}**")
                st.write(f"Training Accuracy: {pipeline_rf.score(X_train, y_train):.2f}")
                st.write(f"Test Accuracy: { pipeline_rf.score(X_test, y_test):.2f}")

            # Display confusion matrix
                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots()
                y_pred = pipeline_lr.predict(X_test)
                ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap='Greens')
                st.pyplot(fig)

