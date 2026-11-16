import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="E-Commerce Analytics Dashboard", page_icon="📊", layout="wide")
sns.set_style("whitegrid")

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "main_data.csv")
    df = pd.read_csv(file_path)
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    return df

df = load_data()

# Sidebar Filter
st.sidebar.header("Filter Data")
min_date = df["order_purchase_timestamp"].min().date()
max_date = df["order_purchase_timestamp"].max().date()

date_range = st.sidebar.date_input(
    "Pilih Rentang Tanggal Pembelian:",
    value=[min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range[0]

filtered_df = df[
    (df["order_purchase_timestamp"].dt.date >= start_date) &
    (df["order_purchase_timestamp"].dt.date <= end_date)
].copy()

# Header Dashboard
st.title("📊 E-Commerce Public Dataset Dashboard (Olist)")
st.markdown("Analisis Performa Kategori Produk, Tren Pesanan, Metode Pembayaran, dan Segmentasi Pelanggan (RFM).")

# KPI Metrics
col1, col2, col3, col4 = st.columns(4)
total_orders = filtered_df["order_id"].nunique()
total_revenue = filtered_df["price"].sum()
total_customers = filtered_df["customer_unique_id"].nunique()
avg_order_value = total_revenue / total_orders if total_orders > 0 else 0

col1.metric("Total Pesanan (Delivered)", f"{total_orders:,}")
col2.metric("Total Pendapatan (BRL)", f"BRL {total_revenue:,.2f}")
col3.metric("Total Pelanggan Unik", f"{total_customers:,}")
col4.metric("Rata-Rata Nilai Pesanan", f"BRL {avg_order_value:,.2f}")

st.markdown("---")

# Visualisasi 1: Top & Bottom Kategori Produk
st.subheader("1. Performa Pendapatan Berdasarkan Kategori Produk")
if len(filtered_df) > 0:
    cat_rev = filtered_df.groupby("product_category_name_english", as_index=False).agg(
        total_revenue=("price", "sum"),
        total_orders=("order_id", "nunique")
    ).sort_values(by="total_revenue", ascending=False)

    fig1, axes1 = plt.subplots(1, 2, figsize=(16, 5))
    top_5 = cat_rev.head(5)
    bottom_5 = cat_rev.tail(5).sort_values(by="total_revenue", ascending=True)

    sns.barplot(x="total_revenue", y="product_category_name_english", data=top_5, hue="product_category_name_english", palette=["#1f77b4"] + ["#c6dbef"]*(len(top_5)-1), legend=False, ax=axes1[0])
    axes1[0].set_title("Top 5 Kategori Pendapatan Tertinggi", fontweight="bold")
    axes1[0].set_xlabel("Pendapatan (BRL)")
    axes1[0].set_ylabel("")
    axes1[0].spines["top"].set_visible(False)
    axes1[0].spines["right"].set_visible(False)

    sns.barplot(x="total_revenue", y="product_category_name_english", data=bottom_5, hue="product_category_name_english", palette=["#d62728"] + ["#fcbba1"]*(len(bottom_5)-1), legend=False, ax=axes1[1])
    axes1[1].set_title("Bottom 5 Kategori Pendapatan Terendah", fontweight="bold")
    axes1[1].set_xlabel("Pendapatan (BRL)")
    axes1[1].set_ylabel("")
    axes1[1].spines["top"].set_visible(False)
    axes1[1].spines["right"].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig1)

st.markdown("---")

# Visualisasi 2: Tren Bulanan & Metode Pembayaran
st.subheader("2. Tren Pesanan Bulanan & Distribusi Metode Pembayaran")
if len(filtered_df) > 0:
    col_left, col_right = st.columns(2)

    with col_left:
        filtered_df["order_month"] = filtered_df["order_purchase_timestamp"].dt.to_period("M").astype(str)
        monthly = filtered_df.groupby("order_month", as_index=False)["order_id"].nunique()
        fig2, ax2 = plt.subplots(figsize=(8, 4.5))
        ax2.plot(monthly["order_month"], monthly["order_id"], marker="o", color="#1f77b4", linewidth=2)
        ax2.set_title("Tren Jumlah Pesanan Bulanan", fontweight="bold")
        ax2.set_xlabel("Bulan")
        ax2.set_ylabel("Jumlah Pesanan")
        ax2.tick_params(axis="x", rotation=45)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        st.pyplot(fig2)

    with col_right:
        pay_dist = filtered_df.drop_duplicates("order_id")["payment_type"].value_counts().reset_index()
        pay_dist.columns = ["payment_type", "count"]
        fig3, ax3 = plt.subplots(figsize=(8, 4.5))
        sns.barplot(x="payment_type", y="count", data=pay_dist, hue="payment_type", palette=["#1f77b4"] + ["#c6dbef"]*(len(pay_dist)-1), legend=False, ax=ax3)
        ax3.set_title("Distribusi Metode Pembayaran Utama", fontweight="bold")
        ax3.set_xlabel("Metode Pembayaran")
        ax3.set_ylabel("Jumlah Pesanan")
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        st.pyplot(fig3)

st.markdown("---")

# Visualisasi 3: Analisis Lanjutan (RFM Analysis)
st.subheader("3. Analisis Lanjutan: RFM Customer Segmentation")
if len(filtered_df) > 0:
    recent_date = filtered_df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
    rfm = filtered_df.groupby("customer_unique_id", as_index=False).agg({
        "order_purchase_timestamp": lambda x: (recent_date - x.max()).days,
        "order_id": "nunique",
        "price": "sum"
    })
    rfm.columns = ["customer_unique_id", "recency", "frequency", "monetary"]

    r_col, f_col, m_col = st.columns(3)
    r_col.metric("Rata-Rata Recency (Hari)", f"{rfm['recency'].mean():.1f} hari")
    f_col.metric("Rata-Rata Frequency", f"{rfm['frequency'].mean():.2f}x")
    m_col.metric("Rata-Rata Monetary", f"BRL {rfm['monetary'].mean():,.2f}")

    fig4, axes4 = plt.subplots(1, 3, figsize=(16, 4.5))
    top_r = rfm.sort_values(by="recency", ascending=True).head(5)
    top_f = rfm.sort_values(by="frequency", ascending=False).head(5)
    top_m = rfm.sort_values(by="monetary", ascending=False).head(5)

    sns.barplot(y=top_r["customer_unique_id"].str[:8], x=top_r["recency"], color="#1f77b4", ax=axes4[0])
    axes4[0].set_title("Top 5 By Recency (Hari Terkini)", fontweight="bold")
    axes4[0].set_xlabel("Recency (Hari)")
    axes4[0].set_ylabel("Customer ID (8-char)")

    sns.barplot(y=top_f["customer_unique_id"].str[:8], x=top_f["frequency"], color="#1f77b4", ax=axes4[1])
    axes4[1].set_title("Top 5 By Frequency (Transaksi)", fontweight="bold")
    axes4[1].set_xlabel("Frequency")
    axes4[1].set_ylabel("")

    sns.barplot(y=top_m["customer_unique_id"].str[:8], x=top_m["monetary"], color="#1f77b4", ax=axes4[2])
    axes4[2].set_title("Top 5 By Monetary (BRL)", fontweight="bold")
    axes4[2].set_xlabel("Monetary (BRL)")
    axes4[2].set_ylabel("")

    plt.tight_layout()
    st.pyplot(fig4)

st.caption("Proyek Akhir Analisis Data dengan Python — Dicoding Indonesia")