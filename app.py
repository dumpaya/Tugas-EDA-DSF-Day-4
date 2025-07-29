import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Pizza Sales Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("pizza_sales.csv")
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce', dayfirst=True)
    df = df.dropna(subset=['order_date'])
    return df

@st.cache_resource
def load_model():
    return load("random_forest_model.joblib")

df = load_data()
model = load_model()

# Sidebar menu
st.sidebar.title("Navigasi")
tab = st.sidebar.radio("Pilih Halaman", ["📈 EDA Global", "🗕 EDA Bulanan", "📈 Prediksi"])

# ==========================
# EDA GLOBAL
# ==========================
if tab == "📈 EDA Global":
    st.title("📈 Exploratory Data Analysis - Global")

    with st.expander("📄 Lihat Data Mentah"):
        st.dataframe(df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎖️ 10 Pizza Terlaris")
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

    st.subheader("💰 Total Revenue per Pizza")
    revenue_per_pizza = df.groupby('pizza_name')['total_price'].sum().sort_values(ascending=False)
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    sns.barplot(x=revenue_per_pizza.values, y=revenue_per_pizza.index, ax=ax3, palette='viridis')
    ax3.set_xlabel("Total Revenue ($)")
    ax3.set_ylabel("Pizza Name")
    st.pyplot(fig3)

    st.subheader("🗓️ Pendapatan Harian")
    daily_revenue = df.groupby('order_date')['total_price'].sum()
    fig4, ax4 = plt.subplots(figsize=(12, 4))
    daily_revenue.plot(ax=ax4)
    ax4.set_title('Total Pendapatan per Hari')
    ax4.set_xlabel('Tanggal')
    ax4.set_ylabel('Total Pendapatan ($)')
    ax4.grid(True)
    st.pyplot(fig4)

    # ============================
    # \ud83d\udcc8 Evaluasi Model Random Forest
    # ============================
    st.subheader("📈 Evaluasi Model Random Forest")

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

    daily['total_quantity'] = daily['quantity']
    daily['avg_unit_price'] = daily['unit_price']
    daily['unique_pizzas'] = daily['pizza_name']
    daily['avg_size'] = daily['pizza_size_num']

    X_model = daily[['day_of_week', 'month', 'day', 'total_quantity', 'avg_unit_price', 'unique_pizzas', 'avg_size']]
    y_model = daily['total_price']

    X_train, X_test, y_train, y_test = train_test_split(X_model, y_model, test_size=0.2, random_state=42)
    X_test = X_test[model.feature_names_in_]
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    st.markdown(f"**R² Score:** `{r2:.4f}`")
    st.markdown(f"**Mean Squared Error:** `{mse:,.2f}`")

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
# EDA BULANAN (Tampilan Visual Seperti Dashboard)
# ==========================
elif tab == "🗕 EDA Bulanan":
    st.title("🗕 EDA Bulanan: Analisis Penjualan Pizza")

    bulan_nama = {
        1: "Januari", 2: "Februari", 3: "Maret", 4: "April",
        5: "Mei", 6: "Juni", 7: "Juli", 8: "Agustus",
        9: "September", 10: "Oktober", 11: "November", 12: "Desember"
    }

    selected_month = st.selectbox("📅 Pilih Bulan", list(bulan_nama.keys()), format_func=lambda x: bulan_nama[x])
    monthly_df = df[df['order_date'].dt.month == selected_month]

    # Data ringkasan
    total_income = monthly_df['total_price'].sum()
    total_orders = monthly_df['order_id'].nunique()
    total_qty = monthly_df['quantity'].sum()
    avg_order_value = total_income / total_orders if total_orders != 0 else 0

    st.markdown("### 🔢 Ringkasan Bulan: " + bulan_nama[selected_month])
    # ======= Layout: 3 Kolom Utama =======
    col1, col2, col3 = st.columns([1, 1, 1])
    
    # === Ringkasan Bulanan (Col 1) ===
    with col1:
        st.metric("💰 Revenue", f"${total_income:,.0f}")
        st.metric("📦 Total Orders", total_orders)
        st.metric("🍕 Total Qty", total_qty)
        st.metric("📊 Avg Order Value", f"${avg_order_value:,.2f}")
    
    # === Pendapatan Harian + Kategori (Col 2) ===
    with col2:
        # Grafik harian
        st.markdown("### 📈 Pendapatan Harian")
        daily_rev = monthly_df.groupby('order_date')['total_price'].sum()
        
        if not daily_rev.empty:
            fig1, ax1 = plt.subplots()
            daily_rev.plot(kind='line', marker='o', ax=ax1)
            ax1.set_title(f"Total Revenue per Hari - {bulan_nama[selected_month]}")
            ax1.set_ylabel("Total Revenue ($)")
            ax1.grid(True)
            st.pyplot(fig1)
        else:
            st.warning("Tidak ada data pendapatan harian untuk bulan ini.")

        st.markdown("### 🥗 Distribusi Kategori Pizza")
        category_dist = monthly_df['pizza_category'].value_counts()
        fig3, ax3 = plt.subplots()
        ax3.pie(category_dist, labels=category_dist.index, autopct='%1.1f%%', startangle=90)
        ax3.axis('equal')
        st.pyplot(fig3)
    
    # === Top Pizza + Ukuran (Col 3) ===
    with col3:
        st.markdown("### 🍕 Top 5 Pizza Terlaris")
        top_pizza = monthly_df['pizza_name'].value_counts().head(5)
        fig2, ax2 = plt.subplots()
        sns.barplot(x=top_pizza.values, y=top_pizza.index, palette="autumn", ax=ax2)
        ax2.set_xlabel("Jumlah Terjual")
        ax2.set_title("Top 5 Pizza")
        st.pyplot(fig2)
    
        st.markdown("### 📏 Distribusi Ukuran Pizza")
        size_dist = monthly_df['pizza_size'].value_counts().sort_index()
        fig4, ax4 = plt.subplots()
        sns.barplot(x=size_dist.index, y=size_dist.values, palette="Blues", ax=ax4)
        ax4.set_ylabel("Jumlah Terjual")
        ax4.set_xlabel("Ukuran Pizza")
        ax4.set_title("Penjualan per Ukuran")
        st.pyplot(fig4)

    # ==========================
    # 🔥 Peak Hours Bulanan (10:00 AM – 9:00 PM)
    # ==========================
    # === Peak Hours (Gabung Col2 + Col3 Seperti Merge & Center) ===
    with st.container():
        col_spacer1, col_merge, col_spacer2 = st.columns([3, 6, 0.1])
        with col_merge:
            st.markdown("### 🔥 Peak Hours Bulanan (11:00 AM – 10:00 PM)")
    
            # --- Gunakan kode peak hour yang sudah benar ---
            df['order_date'] = pd.to_datetime(df['order_date'])
    
            monthly_df = df[df['order_date'].dt.month == selected_month].copy()
    
            monthly_df['order_datetime'] = pd.to_datetime(
                monthly_df['order_date'].dt.date.astype(str) + ' ' + monthly_df['order_time'].astype(str)
            )
    
            monthly_df['hour'] = monthly_df['order_datetime'].dt.hour
            monthly_df['day_name'] = monthly_df['order_datetime'].dt.day_name()
    
            jam_filter = monthly_df[(monthly_df['hour'] >= 10) & (monthly_df['hour'] <= 22)]
    
            pivot_table = jam_filter.pivot_table(
                index='day_name',
                columns='hour',
                values='quantity',
                aggfunc='sum',
                fill_value=0
            )
    
            ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            pivot_table = pivot_table.reindex(ordered_days)
            pivot_table = pivot_table.reindex(columns=range(11, 23), fill_value=0)
    
            pivot_table['Total'] = pivot_table.sum(axis=1)
    
            total_row = pivot_table.sum(numeric_only=True).to_frame().T
            total_row.index = ['Total']
            pivot_table = pd.concat([pivot_table, total_row])
    
            st.dataframe(
                pivot_table.style
                .background_gradient(cmap='YlOrRd', axis=None)
                .format(precision=0),
                use_container_width=True
            )
    
# ==========================
# PREDIKSI PENJUALAN
# ==========================
elif tab == "📈 Prediksi":
    st.title("📈 Prediksi Total Penjualan Harian")

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

# Footer
st.markdown("---")
st.markdown("📦 Dibuat oleh Alya Siti Fathimah • 🚀 Dibimbing.id • `Streamlit + Scikit-learn + EDA`")
