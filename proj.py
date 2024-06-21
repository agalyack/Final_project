import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import streamlit as st
from mlxtend.frequent_patterns import apriori,fpgrowth,association_rules
import sklearn
import plotly.express as px
import matplotlib.pyplot as plt

print(sklearn.__version__)
st.set_page_config(
    page_title="Final Project Machine Learning",
    layout="wide",
    initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    .header-container {
        background-color: purple; /* Purple background */
        padding: 20px; /* Increased padding */
        border-radius: 15px; /* Rounded corners */
        width: 100%; /* Make the container broader */
        margin-left: 10px; /* Align to left margin */
        margin-bottom: 20px; /* Reduce gap between header and subheader */
    }
    .header-text {
        font-family: Arial, sans-serif;
        color: white; /* White text color */
        margin: 0; /* Remove default margin */
        font-size: 24px; /* Increase font size */
    }
    </style>
    """,
    unsafe_allow_html=True
)    
st.markdown('<div class="header-container"><p class="header-text">Final Project Machine learning</p></div>', unsafe_allow_html=True)
with open("drfc_model.pkl","rb") as rfc:
    rf_model_new=pickle.load(rfc)
with open("dlab1.pkl","rb") as dlb1:
    lab_new1=pickle.load(dlb1)
with open("dlab2.pkl","rb") as dlb2:
    lab_new2=pickle.load(dlb2)
with open("dlab3.pkl","rb") as dlb3:
    lab_new3=pickle.load(dlb3)
income_options=['Working','Commercial associate','Pensioner','State servant','Unemployed','Student','Maternity leave']
housingtype_options=['House / apartment','With parents','Municipal apartment','Rented apartment','Office apartment','Co-op apartment']
loanpurpose_options=['XAP','XNA','Repairs','Other','Urgent needs','Buying a used car','Building a house or an annex','Everyday expenses',
                     'Medicine','Payments on other loans','Education','Journey','Purchase of electronic equipment','Buying a new car',
                     'Wedding / gift / holiday','Buying a home','Car repairs','Furniture','Buying a holiday home / land','Business development',
                     'Gasification / water supply','Buying a garage','Hobby','Money for a third person','Refusal to name the goal']
# Sidebar title
st.sidebar.title('Project Demo')
# Profile picture
#st.sidebar.image("profile_picture.jpg", use_column_width=True)

# Name and title
st.sidebar.header("Agalya")
st.sidebar.subheader("Data Scientist")

option = st.sidebar.radio(
    "Choose an option:",
    ("Show Data", "EDA and Insights","Feature Importance","Predicting Bank Defaulters","Product Recommendation System")
)

if option == "Show Data":
    st.write("Showing Data")
    data=pd.read_csv("myfinalloan.csv")
    st.dataframe(data.head(10))
    data1 = {
    "Algorithm": ["Logistic Regression", "Random Forest"],
    "Accuracy": [57, 99],
    "Precision": [56, 99],
    "Recall": [61, 100],
    "F1score":[59,99]
}
    data1_df=pd.DataFrame(data1)
    st.write("Algorithm Performance Metrics")
    st.table(data1_df)
    st.caption("Selected Model: RANDOM FOREST ALGORITHM")
elif option == "EDA and Insights":
    st.write("Showing Analysed")
    df=pd.read_csv("myfinalloan.csv")
    col21,col22=st.columns(2)
    with col21:
        df['AGE'] = df['DAYS_BIRTH'] / 365
        # Create age bins
        age_bins = [20, 30, 40, 50, 60, 70, 80]
        df['AGE_GROUP'] = pd.cut(df['AGE'], bins=age_bins)
        # Convert intervals to strings
        df['AGE_GROUP'] = df['AGE_GROUP'].astype(str)
        age_target_counts = df.groupby(['AGE_GROUP', 'TARGET']).size().reset_index(name='counts')

       # Create a Plotly bar chart
        fig = px.bar(age_target_counts, x='AGE_GROUP', y='counts', color='TARGET', barmode='group',
             title='Distribution of Defaulters by Age Group',
             labels={'AGE_GROUP': 'Age Group (Years)', 'counts': 'Count', 'TARGET': 'Target'})
        st.title('Age Group vs Target Distribution')
        st.plotly_chart(fig)
    col31,col32=st.columns(2)   
    with col31:
        # Group by NAME_INCOME_TYPE and TARGET and count occurrences
        income_target_counts = df.groupby(['NAME_INCOME_TYPE', 'TARGET']).size().reset_index(name='counts')

        # Create separate dataframes for each TARGET value
        df_target_0 = income_target_counts[income_target_counts['TARGET'] == 0]
        df_target_1 = income_target_counts[income_target_counts['TARGET'] == 1]

       # Create a Plotly pie chart for TARGET = 0
        fig_target_0 = px.pie(df_target_0, names='NAME_INCOME_TYPE', values='counts', title='Distribution of Income Type for Defaulters')
        
        st.title('Income Type vs Target Distribution')
 
       # Display the pie charts in Streamlit
        st.subheader('Defaulters')
        st.plotly_chart(fig_target_0)
    with col32:
        st.title('Income Type vs Target Distribution')
       # Create a Plotly pie chart for TARGET = 1
        fig_target_1 = px.pie(df_target_1, names='NAME_INCOME_TYPE', values='counts', title='Distribution of Income Type for NON Defaulters')
        st.subheader('NON Defaulters')
        st.plotly_chart(fig_target_1)
    col41,col42=st.columns(2)
    with col41:
        df_grouped = df.groupby('NAME_CASH_LOAN_PURPOSE')['AMT_CREDIT_x'].sum().reset_index()

        # Create a horizontal bar plot with Plotly
        fig = px.bar(df_grouped, 
             x='AMT_CREDIT_x', 
             y='NAME_CASH_LOAN_PURPOSE', 
             orientation='h', 
             color='NAME_CASH_LOAN_PURPOSE',
             title='Total Amount of Credit by Loan Purpose',
             labels={'AMT_CREDIT_x': 'Total Amount of Credit', 'NAME_CASH_LOAN_PURPOSE': 'Loan Purpose'})
        st.plotly_chart(fig)
    with col42:
        df_grouped = df.groupby('NAME_INCOME_TYPE').agg({
        'AMT_INCOME_TOTAL': 'mean',
        'AMT_CREDIT_x': 'sum'
           }).reset_index()

        # Create a bubble plot with Plotly
        fig = px.scatter(df_grouped, 
                 x='AMT_INCOME_TOTAL', 
                 y='AMT_CREDIT_x', 
                 size='AMT_CREDIT_x', 
                 color='NAME_INCOME_TYPE',
                 title='Credit Amount vs Income Vs Income Type',
                 labels={'AMT_INCOME_TOTAL': 'Total Income', 'AMT_CREDIT_x': 'Total Amount of Credit', 'NAME_INCOME_TYPE': 'Income Type'},
                 size_max=60)  # Adjust size_max to control the maximum bubble size
        st.plotly_chart(fig)
   
    data1=pd.read_csv("myfinalloan_numbers.csv")
    data_corr = data1.corr()
    st.title("Correlation Heatmap")
    # Plot the correlation heatmap
    fig, ax = plt.subplots(figsize=(21, 6))
    sns.heatmap(data_corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    col51,col52=st.columns(2)
    with col51:
        st.title("Box Plot with Outlier Detection-Loan Credit Amount")
        # Create a box plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.boxplot(data1['AMT_CREDIT_x'], vert=False, patch_artist=True, 
           boxprops=dict(facecolor='lightblue'), 
           medianprops=dict(color='red'))

        ax.set_title('Box Plot for AMT_CREDIT_x')
        ax.set_xlabel('Credit Amount')
        ax.grid(True)
        st.pyplot(fig)
    with col52:
        st.title("Box Plot with Outlier Detection-Total Income Amount")
        # Create a box plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.boxplot(data1['AMT_INCOME_TOTAL'], vert=False, patch_artist=True, 
           boxprops=dict(facecolor='purple'), 
           medianprops=dict(color='red'))

        ax.set_title('Box Plot for AMT_INCOME_TOTAL')
        ax.set_xlabel('Total Income')
        ax.grid(True)
        st.pyplot(fig)
elif option=="Feature Importance":
    st.title("Showing Feature Importance")
    data1=pd.read_csv("myfinalloan_numbers.csv")
    data2=data1.drop('TARGET',axis=1)
    # Extract feature importances and features
    feature_importances = rf_model_new.feature_importances_
    features = data2.columns
    indices = np.argsort(feature_importances)[::-1]
# Function to plot feature importances
    def plot_feature_importances():
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances For Target")
        plt.bar(range(data2.shape[1]), feature_importances[indices], align="center")
        plt.xticks(range(data2.shape[1]), [features[i] for i in indices], rotation=90)
        plt.tight_layout()
        st.pyplot(plt)
    plot_feature_importances()

elif option == "Predicting Bank Defaulters":
    st.write("Default or Not")
    st.write("Predicting Bank Defaulters")
    col1,col2,col3,col4,col5=st.columns(5)
    with col1:
        SK_ID_CURR = st.text_input("Customer ID")
        AMT_INCOME_TOTAL = st.text_input("Total Income")
    with col2:
        AMT_CREDIT_x= st.text_input("Credit Amount")
        AMT_GOODS_PRICE_x = st.text_input("Goods Price")
    with col3:
        NAME_INCOME_TYPE = st.selectbox("Income Type", income_options)
        NAME_HOUSING_TYPE= st.selectbox("House Type", housingtype_options)
    with col4:
        DAYS_BIRTH = st.text_input("Days Birth")
        DAYS_EMPLOYED = st.text_input("Number of Days Employed")
    with col5:
        NAME_CASH_LOAN_PURPOSE = st.selectbox("Loan Purpose", loanpurpose_options)

    submitt=st.button("Predict Defaulter")
    if submitt:
        columns=['SK_ID_CURR','AMT_INCOME_TOTAL','AMT_CREDIT_x','AMT_GOODS_PRICE_x','NAME_INCOME_TYPE','NAME_HOUSING_TYPE','DAYS_BIRTH','DAYS_EMPLOYED','NAME_CASH_LOAN_PURPOSE']
        sample_data={
                'SK_ID_CURR': [SK_ID_CURR],
                'AMT_INCOME_TOTAL':[AMT_INCOME_TOTAL],
                'AMT_CREDIT_x': [AMT_CREDIT_x],
                'AMT_GOODS_PRICE_x': [AMT_GOODS_PRICE_x],
                'NAME_INCOME_TYPE': [NAME_INCOME_TYPE],
                'NAME_HOUSING_TYPE': [NAME_HOUSING_TYPE],
                'DAYS_BIRTH': [DAYS_BIRTH],
                'DAYS_EMPLOYED': [DAYS_EMPLOYED],
                'NAME_CASH_LOAN_PURPOSE': [NAME_CASH_LOAN_PURPOSE]
            }   
        df=pd.DataFrame(sample_data,columns=columns)
        df['NAME_INCOME_TYPE']=lab_new1.transform(df['NAME_INCOME_TYPE'])
        df['NAME_HOUSING_TYPE']=lab_new2.transform(df['NAME_HOUSING_TYPE'])
        df['NAME_CASH_LOAN_PURPOSE']=lab_new3.transform(df['NAME_CASH_LOAN_PURPOSE'])
        pred=rf_model_new.predict(df)
        if(pred==1):
            st.markdown(f'### <div class="center-text">Predicted Status = Not a Defaulter</div>', unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(f'### <div class="center-text">Predicted Status = Defaulter</div>', unsafe_allow_html=True)
            st.snow()
elif option=="Product Recommendation System":
    st.header("Product Recommendation")
    with st.container(height=500,border=False):
        st.image('C:/Users/ABI/Desktop/Final/shopping.jpg',caption='Product Recommendation System')
    
    data=pd.read_csv("Groceries data.csv")
    grocery_list=['bottled water','other vegetables','whole milk','citrus fruit','rolls/buns','root vegetables','sausage','soda',
                  'tropical fruit','yogurt','pastry','shopping bags']
    unique_data=data.groupby(['Member_number','itemDescription']).size().unstack().reset_index().fillna(0).set_index('Member_number')
    unique_data = unique_data.map(lambda x: 1 if x > 0 else 0).astype(bool)
    # Apply Apriori algorithm
    frequent_itemsets = apriori(unique_data, min_support=0.09, use_colnames=True)
    
    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="lift")
    rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x))
    rules['consequents'] = rules['consequents'].apply(lambda x: list(x))
    #print(rules)
    def recommend_item(rules_df, input_item, top_n=3):
       input_item_list = [input_item]
       recommendations = rules_df[rules_df['antecedents'].apply(lambda x:x==input_item_list)]
       recommendations = recommendations.sort_values(by='lift', ascending=False)
       recommendation_items = recommendations['consequents'].apply(lambda x: ', '.join(x)).tolist()
       return recommendation_items
    input_item=st.selectbox("Select Product",grocery_list)
    #input_item = 'soda'
    recommendations = recommend_item(rules, input_item)
    st.write(f"Recommendations for item {input_item}:\n", recommendations)
    #print(f"Recommendations for item {input_item}:\n", recommendations)

    
   
        
    


    


 
    
    