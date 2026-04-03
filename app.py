import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt

st.title("💰 Smart Expense Tracker (Income + Expense)")

FILE = "expenses.csv"

# Load or create data
if os.path.exists(FILE):
    df = pd.read_csv(FILE)
else:
    df = pd.DataFrame(columns=["Type", "Amount", "Description", "Category"])

# Sample training data
train_data = [
    ("pizza burger food", "Food"),
    ("bus train travel", "Travel"),
    ("electricity bill current", "Bills"),
    ("movie entertainment", "Entertainment"),
    ("groceries vegetables", "Food"),
]

texts = [x[0] for x in train_data]
labels = [x[1] for x in train_data]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X, labels)

# Input section
type_option = st.selectbox("Select Type", ["Income", "Expense"])
amount = st.number_input("Enter Amount", min_value=0.0)
desc = st.text_input("Enter Description")

if st.button("Add Entry"):
    if desc != "" and amount > 0:
        vec = vectorizer.transform([desc])
        category = model.predict(vec)[0]

        new_data = pd.DataFrame([[type_option, amount, desc, category]],
                                columns=["Type", "Amount", "Description", "Category"])

        df = pd.concat([df, new_data], ignore_index=True)
        df.to_csv(FILE, index=False)

        st.success(f"{type_option} added under category: {category}")

# Show data
st.subheader("📊 Transactions")
st.write(df)

# Calculations
if not df.empty:
    income = df[df["Type"] == "Income"]["Amount"].sum()
    expense = df[df["Type"] == "Expense"]["Amount"].sum()
    balance = income - expense

    st.subheader("💰 Summary")
    st.write(f"Total Income: ₹{income}")
    st.write(f"Total Expense: ₹{expense}")
    st.write(f"Balance: ₹{balance}")

    # Bar chart
    st.subheader("📊 Income vs Expense")
    chart_data = pd.Series([income, expense], index=["Income", "Expense"])

    fig, ax = plt.subplots()
    chart_data.plot(kind="bar", ax=ax)
    st.pyplot(fig)

    # Category spending (only expenses)
    st.subheader("📉 Expense by Category")
    exp_df = df[df["Type"] == "Expense"]

    if not exp_df.empty:
        cat_data = exp_df.groupby("Category")["Amount"].sum()

        fig2, ax2 = plt.subplots()
        cat_data.plot(kind="bar", ax=ax2)
        st.pyplot(fig2)

        # AI Suggestion
        max_cat = cat_data.idxmax()
        st.warning(f"You are spending most on {max_cat}. Try to reduce it!")
