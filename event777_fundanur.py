# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 16:54:20 2025

@author: FUNDANUR UZUN
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 22:26:43 2024

@author: FUNDANUR UZUN
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, wilcoxon, mannwhitneyu
from io import BytesIO

# Function to calculate abnormal return (AR) and cumulative abnormal return (CAR)
def calculate_ar_car(company_prices, index_prices):
    company_returns = np.log(company_prices[1:] / company_prices[:-1])
    index_returns = np.log(index_prices[1:] / index_prices[:-1])
    ar = company_returns - index_returns
    car = np.cumsum(ar)
    ar_statistic = ar / (np.std(ar) if np.std(ar) != 0 else 1)
    return ar, car, ar_statistic

# Function to perform different statistical tests based on user selection
def test_significance(ar, test_type="t-test"):
    if test_type == "t-test":
        t_stat, p_value = ttest_1samp(ar, 0)
        return t_stat, p_value, "T-Test"
    elif test_type == "wilcoxon":
        t_stat, p_value = wilcoxon(ar)
        return t_stat, p_value, "Wilcoxon Signed-Rank Test"
    elif test_type == "mann-whitney":
        median_value = np.median(ar)
        t_stat, p_value = mannwhitneyu(ar, np.full_like(ar, median_value))
        return t_stat, p_value, "Mann-Whitney U Test"
    else:
        return None, None, "Invalid Test"

# Function to calculate momentum
def calculate_momentum(prices, period=10):
    return prices - prices.shift(period)

# Function to perform simple linear regression using numpy
def perform_numpy_regression(company_returns, index_returns):
    A = np.vstack([index_returns, np.ones(len(index_returns))]).T
    beta, alpha = np.linalg.lstsq(A, company_returns, rcond=None)[0]
    y_pred = alpha + beta * index_returns
    ss_res = np.sum((company_returns - y_pred) ** 2)
    ss_tot = np.sum((company_returns - np.mean(company_returns)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return beta, alpha, r_squared

# Function to generate comments based on CAR values
def generate_comments(car_values, company_name):
    if len(car_values) == 0:
        return "Insufficient data to generate a comment."
    initial_value = car_values[0]
    final_value = car_values[-1]
    trend = final_value - initial_value
    if trend > 0:
        return f"The cumulative abnormal return (CAR) for {company_name} shows a positive trend, indicating a potential positive impact of the event."
    elif trend < 0:
        return f"The cumulative abnormal return (CAR) for {company_name} shows a negative trend, suggesting a potential negative impact of the event."
    else:
        return f"The cumulative abnormal return (CAR) for {company_name} is neutral, indicating no significant impact from the event."

# Function to load Excel data
def load_excel_data(uploaded_file, expected_columns):
    try:
        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file)
            df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
            if all(col in df.columns for col in expected_columns):
                st.success(f"{uploaded_file.name} successfully loaded!")
                return df
            else:
                st.error(f"Expected columns: {expected_columns}. Please check the column names.")
                return None
        return None
    except Exception as e:
        st.error(f"Data loading error: {e}")
        return None

# Function to save multiple DataFrames to an Excel file
def to_excel(multiple_df, sheet_names):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for df, sheet in zip(multiple_df, sheet_names):
            df.to_excel(writer, sheet_name=sheet, index=False)
    output.seek(0)
    return output

# Streamlit UI setup
st.title("Advanced Event Day Analysis with Company Comparison 📊")
st.sidebar.header("Data Upload")
uploaded_company_file = st.sidebar.file_uploader("Upload Company Price Data (.xlsx)", type=["xlsx"])
company_data = load_excel_data(uploaded_company_file, ["date", "company", "company_prices"])
uploaded_index_file = st.sidebar.file_uploader("Upload Index Price Data (.xlsx)", type=["xlsx"])
index_data = load_excel_data(uploaded_index_file, ["date", "index_prices"])
uploaded_ipo_file = st.sidebar.file_uploader("Upload IPO Date Data (.xlsx)", type=["xlsx"])
ipo_data = load_excel_data(uploaded_ipo_file, ["company", "ipo_date"])

# Parameter Selection
st.sidebar.header("Parameter Selection")
before_days = st.sidebar.slider("Days Before Event", 5, 30, 20)
after_days = st.sidebar.slider("Days After Event", 5, 30, 20)
test_type = st.sidebar.selectbox("Choose Statistical Test", ["t-test", "wilcoxon", "mann-whitney"])
momentum_period = st.sidebar.slider("Momentum Period (days)", 1, 30, 10)

# Checking if all data is uploaded
if company_data is not None and index_data is not None and ipo_data is not None:
    st.subheader("Uploaded Data")
    st.write("Company Price Data:")
    st.dataframe(company_data)
    st.write("Index Price Data:")
    st.dataframe(index_data)
    st.write("IPO Date Data:")
    st.dataframe(ipo_data)

    all_results = []
    sheet_names = []
    company_car_dict = {}
    MIN_DATA_POINTS = 5

    for company in ipo_data['company'].unique():
        company_prices = company_data[company_data['company'] == company]
        company_ipo_date = ipo_data[ipo_data['company'] == company]['ipo_date'].values[0]

        # UPDATED: Event window with inclusive IPO day
        event_window = company_prices[
            (company_prices['date'] >= company_ipo_date - pd.Timedelta(days=before_days)) &
            (company_prices['date'] <= company_ipo_date + pd.Timedelta(days=after_days))
        ].sort_values(by="date")

        if len(event_window) < MIN_DATA_POINTS:
            st.warning(f"{company} için yeterli veri bulunamadı. Mevcut veri sayısı: {len(event_window)}")
            continue

        merged_data = pd.merge(event_window, index_data, on="date", how="inner")
        if len(merged_data) > 1:
            st.subheader(f"Abnormal Return Analysis for {company}")
            ar, car, ar_stat = calculate_ar_car(merged_data["company_prices"].values, merged_data["index_prices"].values)
            t_stat, p_value, test_name = test_significance(ar, test_type)
            st.write(f"{company} - {test_name} - T-Statistic: {t_stat:.4f}, P-Value: {p_value:.4f}")

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(range(len(ar)), ar, label='Abnormal Return (AR)', linestyle='-', marker='o')
            ax.plot(range(len(car)), car, label='Cumulative Abnormal Return (CAR)', linestyle='-', marker='x')
            ax.axhline(0, color='gray', linewidth=1)
            ax.set_title(f"AR and CAR for {company}")
            ax.set_xlabel("Days relative to event")
            ax.set_ylabel("Return")
            ax.legend()
            st.pyplot(fig)

            momentum = calculate_momentum(merged_data["company_prices"], period=momentum_period)
            st.write(f"Momentum Analysis for {company}:")
            st.line_chart(momentum, width=700, height=400)

            company_returns = np.log(merged_data["company_prices"].values[1:] / merged_data["company_prices"].values[:-1])
            index_returns = np.log(merged_data["index_prices"].values[1:] / merged_data["index_prices"].values[:-1])
            beta, alpha, r_squared = perform_numpy_regression(company_returns, index_returns)
            st.write(f"Regression Analysis for {company}: Beta = {beta:.4f}, Alpha = {alpha:.4f}, R-squared = {r_squared:.4f}")

            results_df = pd.DataFrame({
                "Date": merged_data["date"].iloc[1:],
                "Abnormal Return (AR)": ar,
                "Cumulative Abnormal Return (CAR)": car,
                "Momentum": momentum.iloc[1:].values,
                "AR Statistic": ar_stat,
                "Beta": [beta] * len(ar),
                "Alpha": [alpha] * len(ar),
                "R-Squared": [r_squared] * len(ar)
            })
            st.dataframe(results_df)

            comment = generate_comments(car, company)
            st.markdown(f"**Automatic Comment:** {comment}")

            all_results.append(results_df)
            sheet_names.append(f"{company} Results")
            company_car_dict[company] = car

    if all_results:
        excel_data = to_excel(all_results, sheet_names)
        st.download_button(label="Download All Analysis Results as Excel", data=excel_data, file_name='all_analysis_results.xlsx')

    st.subheader("Company Comparison Section")
    selected_companies = st.multiselect("Select Companies for Comparison", list(company_car_dict.keys()))

    if selected_companies:
        plt.figure(figsize=(12, 6))
        for company in selected_companies:
            plt.plot(company_car_dict[company], label=f"{company} CAR")
        plt.xlabel("Days")
        plt.ylabel("Cumulative Abnormal Return (CAR)")
        plt.title("Comparison of Selected Companies' Cumulative Abnormal Return (CAR)")
        plt.legend()
        st.pyplot(plt)
        st.markdown("**Automatic Comparison Comment:** Comparing the CAR values of the selected companies shows the relative impact of the event on each company.")
