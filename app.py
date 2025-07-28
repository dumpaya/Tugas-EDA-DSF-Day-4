import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from datetime import datetime

# Atur tampilan
st.set_page_config(page_title="Pizza Sales Dashboard", layout="wide")

# Load dataset dan model
@st.cache_data
def load_data():
    return pd.read_csv("pizza_sales.csv")

@st.cache_resource
def load_model():
    return load("random_forest_model.joblib")

df = load_data()
model = load_model()

# Sidebar
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["📊 EDA", "📈 Prediksi Penjualan"])

# ==========================
# Halaman EDA
# ==========================
if page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis - Pizza Sales")

    # Konversi order_date agar bisa digunakan
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce', dayfirst=True)
    df = df.dropna(subset=['order_date'])  # Drop jika parsing gagal

    # 🗂️ Tampilkan data mentah dalam expander di halaman utama
    with st.expander("📄 Lihat Data Mentah (Raw Data)"):
        st.dataframe(df, use_container_width=True)

    # Visualisasi 1: 10 Pizza Terlaris
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🥇 10 Pizza Terlaris")
        pizza_counts = df['pizza_name'].value_counts().head(10)
        fig1, ax1 = plt.subplots()
        ax1.pie(pizza_counts, labels=pizza_counts.index, autopct='%1.1f%%', startangle=140)
        ax1.axis('equal')
        st.pyplot(fig1)

    with col2:
        st.subheader("🍕 Kategori Pizza Terpopuler")
        pizza_category_counts = df['pizza_category'].value_counts()
        fig2, ax2 = plt.subplots()
        ax2.pie(pizza_category_counts, labels=pizza_category_counts.index, autopct='%1.1f%%', startangle=140)
        ax2.axis('equal')
        st.pyplot(fig2)

    # Visualisasi 2: Revenue per Pizza
    st.subheader("💰 Total Revenue per Pizza")
    revenue_per_pizza = df.groupby('pizza_name')['total_price'].sum().sort_values(ascending=False)
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    sns.barplot(x=revenue_per_pizza.values, y=revenue_per_pizza.index, ax=ax3, palette='viridis')
    ax3.set_xlabel("Total Revenue ($)")
    ax3.set_ylabel("Pizza Name")
    ax3.set_title("Total Revenue per Pizza")
    st.pyplot(fig3)

    # Visualisasi 3: Pendapatan Harian
    st.subheader("📅 Pendapatan Harian")
    daily_revenue = df.groupby('order_date')['total_price'].sum()
    fig4, ax4 = plt.subplots(figsize=(12, 4))
    daily_revenue.plot(ax=ax4)
    ax4.set_title('Total Pendapatan per Hari')
    ax4.set_xlabel('Tanggal')
    ax4.set_ylabel('Total Pendapatan ($)')
    ax4.grid(True)
    st.pyplot(fig4)

    # ============================
    # 🔍 Evaluasi Model Random Forest
    # ============================
    st.subheader("📈 Evaluasi Model Random Forest")

    # Feature engineering untuk model
    df['pizza_size_num'] = df['pizza_size'].map({'S':1,'M':2,'L':3,'XL':4,'XXL':5})
    daily = df.groupby('order_date').agg({
        'quantity': 'sum',
        'unit_price': 'mean',
        'pizza_name': 'nunique',
        'pizza_size_num': 'mean',
        'total_price': 'sum'
    }).reset_index()

    daily['day_of_week'] = daily['order_date'].dt.dayofweek
    daily['month'] = daily['order_date'].dt.month
    daily['day'] = daily['order_date'].dt.day

    # Fitur dan target
    X_model = daily[['day_of_week', 'month', 'day', 'quantity', 'unit_price', 'pizza_name', 'pizza_size_num']]
    y_model = daily['total_price']

    # Split data
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score

    X_train, X_test, y_train, y_test = train_test_split(X_model, y_model, test_size=0.2, random_state=42)

    # Load model
    model = load_model()
    # Pastikan kolom yang dibutuhkan tersedia dan cocok
    daily['total_quantity'] = daily['quantity']
    daily['avg_unit_price'] = daily['unit_price']
    daily['unique_pizzas'] = daily['pizza_name']
    daily['avg_size'] = daily['pizza_size_num']
    
    X_model = daily[['day_of_week', 'month', 'day', 'total_quantity', 'avg_unit_price', 'unique_pizzas', 'avg_size']]


    # Evaluasi
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    # Tampilkan metrik
    st.markdown(f"**R² Score:** `{r2:.4f}`")
    st.markdown(f"**Mean Squared Error:** `{mse:,.2f}`")

    # Visualisasi Prediksi vs Aktual
    st.subheader("📊 Grafik Prediksi vs Aktual")
    fig5, ax5 = plt.subplots(figsize=(10, 4))
    ax5.plot(y_test.values, label='Actual', marker='o')
    ax5.plot(y_pred, label='Predicted', marker='x')
    ax5.set_xlabel("Sample Index")
    ax5.set_ylabel("Total Penjualan ($)")
    ax5.set_title("Prediksi vs Aktual - Random Forest")
    ax5.legend()
    st.pyplot(fig5)

# ==========================
# Halaman Prediksi
# ==========================
elif page == "📈 Prediksi Penjualan":
    st.title("📈 Prediksi Total Penjualan Harian")

    st.markdown("Masukkan data harian untuk memprediksi **total harga penjualan pizza**.")

    with st.form(key="form_prediksi"):
        col1, col2, col3 = st.columns(3)
        with col1:
            day_of_week = st.slider("Hari (0=Senin, ..., 6=Minggu)", 0, 6, 2)
            month = st.selectbox("Bulan", list(range(1, 13)), index=6)
        with col2:
            day = st.slider("Tanggal", 1, 31, 15)
            total_quantity = st.number_input("Total Jumlah Pizza", min_value=1, value=100)
        with col3:
            avg_unit_price = st.number_input("Rata-rata Harga per Unit", min_value=0.0, value=20.0)
            unique_pizzas = st.slider("Jumlah Pizza Unik", 1, 32, 5)
            avg_size = st.slider("Ukuran Rata-Rata Pizza (1=S, ..., 5=XXL)", 1.0, 5.0, 3.0)

        submit = st.form_submit_button("Prediksi Sekarang")

    if submit:
        input_df = pd.DataFrame({
            'day_of_week': [day_of_week],
            'month': [month],
            'day': [day],
            'total_quantity': [total_quantity],
            'avg_unit_price': [avg_unit_price],
            'unique_pizzas': [unique_pizzas],
            'avg_size': [avg_size]
        })

        prediction = model.predict(input_df)[0]
        st.success(f"✅ Total Penjualan yang Diprediksi: **${prediction:,.2f}**")

        st.caption("Model: Random Forest Regressor (tuned)")

# Footer
st.markdown("---")
st.markdown("📦 Dibuat oleh Alya Siti Fathimah • 🚀 Dibimbing.id • `Streamlit + Scikit-learn + EDA`")
