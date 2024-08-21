import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Define cluster descriptions
cluster_descriptions = {
    0: "Cluster 0: Young, low-income individuals who are often students or early in their careers.",
    1: "Cluster 1: Middle-aged, high-income professionals with stable jobs and higher education.",
    2: "Cluster 2: Older individuals with high income, often retirees or high-level executives.",
    3: "Cluster 3: Young adults with moderate income, often single or in the early stages of their careers.",
    # Add descriptions for other clusters as needed
}

st.title('Customer Segmentation Analysis')
st.divider()
st.sidebar.header('Input Data')
sex = st.sidebar.selectbox('Sex', options=['male', 'female'])
marital_status = st.sidebar.selectbox('Marital Status', options=['single', 'non-single'])
age = st.sidebar.number_input('Age', min_value=0, max_value=120, step=1)
education = st.sidebar.selectbox('Education', options=['high school', 'university'])
income = st.sidebar.number_input('Income', min_value=0, step=1)
occupation = st.sidebar.selectbox('Occupation', options=['skilled employee', 'unemployed'])
settlement_size = st.sidebar.selectbox('Settlement Size', options=['big city', 'mid-sized city', 'small city'])

submit_button = st.sidebar.button('Submit')

if submit_button:

    input_data = pd.DataFrame({
        'Sex': [sex],
        'Marital status': [marital_status],
        'Age': [age],
        'Education': [education],
        'Income': [income],
        'Occupation': [occupation],
        'Settlement size': [settlement_size]
    })

    for column, le in label_encoders.items():
        input_data[column] = le.transform(input_data[column])

    input_data[['Age', 'Income']] = scaler.transform(input_data[['Age', 'Income']])
    
    input_data['Cluster'] = kmeans.predict(input_data[['Age', 'Income', 'Sex', 'Marital status', 'Education', 'Occupation', 'Settlement size']])
    cluster_label = input_data["Cluster"].values[0]
    st.header('Cluster Prediction')
    st.markdown(f"""
    <div style="border: 1px solid #ddd; border-radius: 10px; padding: 20px; box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);">
        <h3 style="margin-top: 0;">Prediction Summary</h3>
        <p><strong>Predicted Cluster:</strong> <span style="font-size: 24px; font-weight: bold; color: #4CAF50;">{cluster_label}</span></p>
        <p>{cluster_descriptions.get(cluster_label, "No description available for this cluster.")}</p>
    </div>
    """, unsafe_allow_html=True)

    df = pd.read_csv('Mallcust.csv')
    
    for column, le in label_encoders.items():
        df[column] = le.transform(df[column])
    
    df[['Age', 'Income']] = scaler.transform(df[['Age', 'Income']])
    df['Cluster'] = kmeans.predict(df[['Age', 'Income', 'Sex', 'Marital status', 'Education', 'Occupation', 'Settlement size']])

    df[['Age', 'Income']] = scaler.inverse_transform(df[['Age', 'Income']])
    
    input_data[['Age', 'Income']] = scaler.inverse_transform(input_data[['Age', 'Income']])
    df = pd.concat([df, input_data], ignore_index=True)
    
    st.divider()
    st.header('Clustering Visualization')
    fig = px.scatter(df, x='Age', y='Income', color='Cluster', title='Customer Clusters', 
                     color_discrete_sequence=px.colors.qualitative.Plotly)
    fig.add_trace(go.Scatter(
        x=input_data['Age'],
        y=input_data['Income'],
        mode='markers+text',
        marker=dict(size=12, color='red', symbol='cross', line=dict(color='black', width=2)),
        text=['New Data Point'],
        textposition='top center',
        name='New Data Point'
    ))
    fig.add_annotation(
        x=input_data['Age'].values[0],
        y=input_data['Income'].values[0],
        ax=df[df['Cluster'] == cluster_label]['Age'].mean(),
        ay=df[df['Cluster'] == cluster_label]['Income'].mean(),
        xref='x',
        yref='y',
        axref='x',
        ayref='y',
        text='New Data Point',
        showarrow=True,
        arrowhead=2
    )
    st.plotly_chart(fig)
    

    st.header('3D Cluster Visualization')
    fig = px.scatter_3d(df, x='Age', y='Income', z='Income', color='Cluster', title='3D Visualization of Clusters',
                        color_discrete_sequence=px.colors.qualitative.Plotly)
    fig.add_trace(go.Scatter3d(
        x=input_data['Age'],
        y=input_data['Income'],
        z=input_data['Income'],
        mode='markers+text',
        marker=dict(size=12, color='red', symbol='cross', line=dict(color='black', width=2)),
        text=['New Data Point'],
        textposition='top center',
        name='New Data Point'
    ))
    st.plotly_chart(fig)
