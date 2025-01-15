import pickle
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from openai import OpenAI
import utils as ut
import plotly.express as px
from scipy.stats import percentileofscore


df = pd.read_csv('churn 2.csv')

client = OpenAI(base_url="https://api.groq.com/openai/v1",
                api_key=os.environ.get('GROQ_API_KEY'))


def calculate_percentiles(selected_customer, df):
    percentiles = {}
    for feature in df.columns:
        if feature not in ['CustomerId', 'Surname', 'Exited', "RowNumber", "Geography", "HasCrCard", "IsActiveMember", "Gender"]:  # Exclude non-numeric and target columns
            # Calculate the percentile rank of the selected customer's feature value
            rank = percentileofscore(df[feature], selected_customer[feature])
            percentiles[feature] = rank  # Store the rank directly
    return percentiles


def display_percentiles(percentiles):
    percentile_df = pd.DataFrame(list(percentiles.items()), columns=['Feature', 'Percentile'])
    fig = px.bar(percentile_df, x='Feature', y='Percentile',
                  title="Customer Percentiles",
                  labels={'Percentile': 'Percentile (%)'},
                  text='Percentile')
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='auto')
    fig.update_layout(yaxis=dict(tickvals=np.arange(0, 101, 10)), 
                      yaxis_title='Percentile (%)',
                      xaxis_title='Features',
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      font=dict(color='white'))
    return fig


def explain_prediction(probability, input_dict, surname):
  prompt = f"""
    You are a senior data scientist at a bank specializing in interpreting and explaining predictions of customer churn. Your task is to provide a clear, concise explanation of why a customer named {surname} is at risk of churning ({round(probability * 100, 1)}%) based on their profile.

    ### Customer Information:
    {input_dict}

    ### Top 10 Features Contributing to Churn Prediction:
    | Feature             | Importance |
    |---------------------|------------|
    | NumOfProducts       | 0.323888   |
    | IsActiveMember      | 0.164146   |
    | Age                 | 0.109550   |
    | Geography_Germany   | 0.091373   |
    | Balance             | 0.052786   |
    | Geography_France    | 0.046463   |
    | Gender_Female       | 0.045283   |
    | Geography_Spain     | 0.036855   |
    | CreditScore         | 0.035005   |
    | EstimatedSalary     | 0.032655   |

    ### Summary Statistics:
    **Churned Customers**:  
    {df[df["Exited"] == 1].describe()}

    **Non-Churned Customers**:  
    {df[df["Exited"] == 0].describe()}

    ### Instructions:
    - If the churn risk is over 40%, provide a 3-sentence explanation of why the customer is likely to churn, focusing on key factors from the customer's profile and feature importance.
    - If the churn risk is under 40%, explain in 3 sentences why the customer is less likely to churn, again focusing on relevant customer data and statistics.
    - The explanation should be purely customer-centric, without any mention of the probability, machine learning models, or the top 10 feature list.

    Craft the explanation with a natural, conversational tone that’s easy to understand by non-technical stakeholders.
    """


  raw_response = client.chat.completions.create(
      model="llama-3.3-70b-versatile",
      messages=[{
          "role": "user",
          "content": prompt
      }])

  return raw_response.choices[0].message.content


def generate_email(probability, input_dict, explanation, surname):
  prompt = prompt = f"""
    You are a senior customer retention manager at HS Bank. Your goal is to retain a customer named {surname}, who may be at risk of churning.

    ### Customer Information:
    {input_dict}

    ### Explanation of Churn Risk:
    {explanation}

    ### Task:
    Compose a personalized email to {surname}, encouraging them to remain loyal to HS Bank. Tailor the rewards based on the customer’s specific profile characteristics, and vary the incentives based on the churn risk.

    ### Churn Risk:
    - **If the churn risk is greater than 60%**, emphasize urgent, high-value rewards like waived fees, bonuses, or premium services.
    - **If the churn risk is between 40% and 60%**, offer moderately attractive incentives such as cashback, better savings rates, or rewards on specific products.
    - **If the churn risk is under 40%**, focus on customer appreciation and small loyalty rewards, without being too aggressive in incentives.

    ### Guidelines for Incentives:
    - **High Balance (over $50,000)**: Consider offering access to a **premium financial advisor**, **higher interest rates** on savings, or **personalized wealth management services**.
    - **Low Balance (under $5,000)**: Offer rewards like **waived overdraft fees**, a **$50 deposit match**, or **free checkbooks** for a year.
    - **High Tenure (5+ years)**: Mention their loyalty and offer **exclusive access to a new product** or a **loyalty bonus** such as **double reward points** for the next 3 months.
    - **Active Credit Card Users**: Consider incentives like **5% cashback** on credit card purchases for the next 6 months, or **increased rewards on travel purchases**.
    - **Customers with Dormant Products**: Offer a **$50 bonus** for reactivating a dormant account or waiving the **annual fee** for a year on their credit card.

    ### Tone and Messaging:
    - **High Churn Risk**: Be more persuasive and emphasize the urgency of retaining their relationship with the bank.
    - **Moderate Churn Risk**: Focus on long-term benefits and encourage them to continue using your services with moderate rewards.
    - **Low Churn Risk**: Express appreciation for their loyalty, emphasizing their value and offering a small token of appreciation.

    ### Example Structure of the Email:
    - **Opening**: Express appreciation for their loyalty and mention how valued they are as a customer.
    - **Middle**: Subtly address their current relationship with the bank without directly referencing their product usage (e.g., “As part of our ongoing commitment to customers, we’re extending some special offers tailored for you...”).
    - **Incentives**: Present the incentives based on their profile with clear numerical values and use bullet points for clarity.

    ### Sample Email Incentive Section:
    - **For High-balance, High-risk customers**: 
      - Access to **premium financial advisory services** to help grow your wealth.
      - **Waived monthly fees** for the next year on all savings accounts.
      - **0.75% interest rate boost** on savings for balances over $50,000.
    - **For Low-balance, Low-risk customers**: 
      - **$50 deposit bonus** when you increase your savings account balance by $500 in the next 3 months.
      - **Free overdraft protection** for the next year.
      - **Free checks** for your personal checking account.
    - **For Credit Card Users**:
      - **5% cashback** on all purchases for the next 6 months.
      - **Double rewards points** on all travel purchases until year-end.
    """



  raw_response = client.chat.completions.create(
      model="llama-3.3-70b-versatile",
      messages=[{
          "role": "user",
          "content": prompt
      }])

  print("\n\n EMAIL PROMPT", prompt)
  return raw_response.choices[0].message.content


def load_model(filename):
  with open(filename, "rb") as file:
    return pickle.load(file)


xgboost_model = load_model("xgb_model.pkl")

naive_bayes_model = load_model("nb_model.pkl")

random_forest_model = load_model("rf_model.pkl")

decision_tree_model = load_model("dt_model.pkl")

svm_model = load_model("svm_model.pkl")

knn_model = load_model("knn_model.pkl")

voting_classer_model = load_model("voting_hard_clf.pkl")

xgboost_SMOTE_model = load_model("xgboost-SMOTE.pkl")

xgboost_featureEngineered_model = load_model("xgboost-featureEngineered.pkl")

gb_model = load_model("gb_model.pkl")

stacking_model = load_model("stacking_model.pkl")


def prepare_input(credit_score, location, gender, age, tenure, balance,
                  num_products, has_credit_card, is_active_member,
                  estimated_salary):
  input_dict = {
      "CreditScore": credit_score,
      "Age": age,
      "Tenure": tenure,
      "Balance": balance,
      "NumOfProducts": num_products,
      "HasCrCard": int(has_credit_card),
      "IsActiveMember": int(is_active_member),
      "EstimatedSalary": estimated_salary,
      "Geography_France": 1 if location == "France" else 0,
      "Geography_Germany": 1 if location == "Germany" else 0,
      "Geography_Spain": 1 if location == "Spain" else 0,
      "Gender_Male": 1 if gender == "Male" else 0,
      "Gender_Female": 1 if gender == "Female" else 0,
  }
  input_df = pd.DataFrame([input_dict])
  return input_df, input_dict

def make_predictions(input_df, input_dict):

  probabilities = {
      "XGBoost": xgboost_model.predict_proba(input_df)[0][1],
      "Random Forest": random_forest_model.predict_proba(input_df)[0][1],
      "K-Nearest Neighbors": knn_model.predict_proba(input_df)[0][1],
      "Gradient Boosting": gb_model.predict_proba(input_df)[0][1],
      "Stacking Classifer": stacking_model.predict_proba(input_df)[0][1],
  }
  avg_probability = np.mean(list(probabilities.values()))

  col1, col2 = st.columns(2)
    
  with col1:
    fig = ut.create_gauge_chart(avg_probability)
    st.plotly_chart(fig, use_container_width=True)
    st.write(
        f"The customer has a {avg_probability:.2%} probability of churning.")

  with col2:
    fig_probs = ut.create_model_probability_chart(probabilities)
    st.plotly_chart(fig_probs, use_container_width=True)
      
  percentiles = calculate_percentiles(selected_customer, df)
  percentile_fig = display_percentiles(percentiles)
  st.plotly_chart(percentile_fig, use_container_width=True)

  return avg_probability

st.title("Customer Churn Prediction")

# Upload the dataset
df = pd.read_csv("churn 2.csv")

# Create list of customers aka dropdown
customers = [
    f"{row['CustomerId']} - {row['Surname']}" for index, row in df.iterrows()
]

# Create a dropdown for customer selection
selected_customer_option = st.selectbox("Select a customer", customers)

# When a customer is selected store the customer id and surname into diff variables
if selected_customer_option:
  selected_customer_id = int(selected_customer_option.split(" - ")[0])

  print("Selected Customer ID:", selected_customer_id)

  selected_surname = selected_customer_option.split(" - ")[1]

  print("Surname:", selected_surname)
  # Finds row that matches customer id of selected customer
  selected_customer = df.loc[df['CustomerId'] == selected_customer_id].iloc[0]
  print("Selected Cusomter:", selected_customer)
  

  # Creates two columns in the variables
  col1, col2 = st.columns(2)
  # Displays the selected customer's details
  with col1:

    credit_score = st.number_input("Credit Score",
                                   min_value=300,
                                   max_value=850,
                                   value=int(selected_customer['CreditScore']))

    location = st.selectbox("Location", ["Spain", "France", "Germany"],
                            index=["Spain", "France", "Germany"
                                   ].index(selected_customer["Geography"]))

    gender = st.radio("Gender", ["Male", "Female"],
                      index=0 if selected_customer["Gender"] == "Male" else 1)

    age = st.number_input("Age",
                          min_value=10,
                          max_value=100,
                          value=int(selected_customer["Age"]))

    tenure = st.number_input("Tenure",
                             min_value=0,
                             max_value=50,
                             value=int(selected_customer["Tenure"]))

  with col2:
    balance = st.number_input("Balance",
                              min_value=0.0,
                              value=float(selected_customer["Balance"]))

    num_products = st.number_input("Number of Products",
                                   min_value=1,
                                   max_value=10,
                                   value=int(
                                       selected_customer["NumOfProducts"]))

    has_credit_card = st.checkbox("Has Credit Card",
                                  value=bool(selected_customer["HasCrCard"]))

    is_active_member = st.checkbox("Is Active Member",
                                   value=bool(
                                       selected_customer["IsActiveMember"]))

    estimated_salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        value=float(selected_customer["EstimatedSalary"]))
      
  if st.button("Predict Churn"): 

      input_df, input_dict = prepare_input(credit_score, location, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary)
      avg_probability = make_predictions(input_df, input_dict)
      explanation = explain_prediction(avg_probability, input_dict,
                                           selected_customer["Surname"])
    
      st.write("---")
      st.write("Explanation of Prediction")
      st.write(explanation)
    
      email = generate_email(avg_probability, input_dict, explanation,
                             selected_customer["Surname"])
      st.write("---")
      st.write("Personalized Email")
      st.write(email)
  
