import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Wine Quality EDA Dashboard",
    page_icon="🍷",
    layout="wide"
)

plt.rcParams["figure.autolayout"] = True

# ================= LOAD DATA =================
df = pd.read_csv(os.path.join("data", "winequality.csv"))

# ================= BASIC PREP =================
df["quality_label"] = np.where(df["quality"] >= 7, "Good Wine", "Bad Wine")

num_cols = df.select_dtypes(include=np.number).columns.tolist()
num_features = [col for col in num_cols if col != "quality"]
cat_cols = ["quality_label"]

# ================= TITLE =================
st.markdown(
    "<h1 style='text-align:center;'>🍷 Wine Quality EDA Dashboard</h1>",
    unsafe_allow_html=True
)


# ================= SIDEBAR =================
st.sidebar.header("🛠️ Analysis Controls")

analysis_type = st.sidebar.selectbox(
    "📌 Select Analysis Type",
    ["Univariate Analysis", "Bivariate Analysis"]
)

# ================= KPI SECTION =================
st.subheader("📌 Dataset Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("🍾 Total Wines", df.shape[0])
c2.metric("🍷 Avg Alcohol (%)", round(df["alcohol"].mean(), 2))
c3.metric("⭐ Avg Quality", round(df["quality"].mean(), 2))
c4.metric(
    "✅ Good Wine (%)",
    round((df[df["quality"] >= 7].shape[0] / df.shape[0]) * 100, 2)
)
st.markdown("---")

# ================= UNIVARIATE ANALYSIS =================
if analysis_type == "Univariate Analysis":

    st.subheader("📊 Univariate Analysis")

    feature = st.sidebar.selectbox(
        "🔍 Select Numerical Feature",
        num_features + ["quality"]
    )

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df[feature], kde=True, bins=30, ax=ax)
        ax.set_title(f"Distribution of {feature}")
        st.pyplot(fig, use_container_width=True)

        st.caption(
            f"📈 This chart shows how **{feature}** values are distributed across wines. "
            "It helps identify common ranges and overall shape of the data."
        )

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x=df[feature], ax=ax)
        ax.set_title(f"Spread of {feature}")
        st.pyplot(fig, use_container_width=True)

        st.caption(
            f"📦 This boxplot highlights the typical range, median, and possible outliers "
            f"for **{feature}**."
        )

    st.info(
        "🧠 Univariate analysis helps understand individual features "
        "before exploring relationships between them."
    )

# ================= BIVARIATE ANALYSIS =================
else:

    st.subheader("📈 Bivariate Analysis")

    bi_type = st.sidebar.selectbox(
        "🔗 Select Bivariate Type",
        ["Num vs Num", "Num vs Cat", "Cat vs Cat"]
    )

    # -------- Num vs Num --------
    if bi_type == "Num vs Num":

        x = st.sidebar.selectbox("X-axis (Numerical)", num_features)
        y = st.sidebar.selectbox("Y-axis (Numerical)", num_features, index=1)

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.scatterplot(x=df[x], y=df[y], alpha=0.6, ax=ax)
        ax.set_title(f"{x} vs {y}")
        st.pyplot(fig, use_container_width=True)

        st.caption(
            f"🔍 Each dot represents a wine. This plot helps observe how **{x}** "
            f"changes with **{y}**, and whether a relationship exists."
        )

        st.info(
            "📌 Scatter plots are useful for identifying trends or correlations "
            "between numerical features."
        )

    # -------- Num vs Cat --------
    elif bi_type == "Num vs Cat":

        num = st.sidebar.selectbox("Numerical Feature", num_features)
        cat = st.sidebar.selectbox("Categorical Feature", cat_cols)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(
            x=df[cat],
            y=df[num],
            errorbar=None,
            ax=ax
        )
        ax.set_title(f"Average {num} by {cat}")
        ax.set_xlabel("")
        ax.set_ylabel(f"Mean {num}")
        st.pyplot(fig, use_container_width=True)

        st.caption(
            f"📊 This bar chart compares the **average {num}** between "
            "Good and Bad quality wines."
        )

        st.info(
            "📌 Bar charts are ideal for comparing average values across categories."
        )

    # -------- Cat vs Cat --------
    else:

        x = st.sidebar.selectbox("X-axis (Category)", cat_cols)
        y = st.sidebar.selectbox("Hue (Category)", cat_cols)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x=df[x], hue=df[y], ax=ax)

        ax.set_title(f"{x} vs {y}")
        ax.set_xlabel(x)
        ax.set_ylabel("Count")

        for container in ax.containers:
            ax.bar_label(container, label_type="edge", fontsize=9)

        st.pyplot(fig, use_container_width=True)

        st.caption(
            "📊 This chart shows how many wines fall into each quality category. "
            "Numbers on bars represent exact counts."
        )

        st.info(
            "📌 Count plots help understand class balance and category distribution."
        )
feature_cols = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol"
]


# ================= FEATURE CORRELATION =================
st.subheader("🏆 Feature Correlation with Quality")

corr_df = (
    df[feature_cols]
    .corrwith(df["quality"])
    .reset_index()
)

corr_df.columns = ["Feature", "Correlation"]
corr_df["Abs_Correlation"] = corr_df["Correlation"].abs()
corr_df = corr_df.sort_values("Abs_Correlation", ascending=False)

c1, c2 = st.columns(2)

with c1:
    st.dataframe(
        corr_df[["Feature", "Correlation"]],
        use_container_width=True
    )

with c2:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        x="Abs_Correlation",
        y="Feature",
        data=corr_df,
        ax=ax
    )
    ax.set_title("Absolute Correlation with Quality")
    st.pyplot(fig, use_container_width=True)

st.warning(
    "No single feature strongly correlates with wine quality. "
    "This confirms that quality depends on multiple interacting features."
)
