import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt

st.title("💰 Smart Expense Tracker")

FILE = "expenses.csv"

# Load or create data
if os.path.exists(FILE):
    df = pd.read_csv(FILE)
else:
    df = pd.DataFrame(columns=["Amount", "Description", "Category"])

# Sample training data for AI
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
amount = st.number_input("Enter Amount")
desc = st.text_input("Enter Description")

if st.button("Add Expense"):
    if desc != "":
        vec = vectorizer.transform([desc])
        category = model.predict(vec)[0]

        new_data = pd.DataFrame([[amount, desc, category]],
                                columns=["Amount", "Description", "Category"])

        df = pd.concat([df, new_data], ignore_index=True)
        df.to_csv(FILE, index=False)

        st.success(f"Added under category: {category}")

# Show data
st.subheader("📊 Expense Data")
st.write(df)

# Chart
if not df.empty:
    st.subheader("Spending by Category")

    chart_data = df.groupby("Category")["Amount"].sum()

    fig, ax = plt.subplots()
    chart_data.plot(kind="bar", ax=ax)
    st.pyplot(fig)

    # AI Suggestion
    max_cat = chart_data.idxmax()
    st.warning(f"You are spending most on: {max_cat}. Try to reduce it!")
