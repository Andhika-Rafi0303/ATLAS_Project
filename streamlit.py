import streamlit as st
import pandas as pd
import pickle
import altair as alt
from io import BytesIO

st.set_page_config(
    page_title="ATLAS",
    layout="wide"  # Set layout to wide mode
)

pipe = pickle.load(open('model.pkl', 'rb'))
model_cost = pickle.load(open('Cost_xgbrf_model.pkl', 'rb'))
st.html("styles.html")

# Sidebar content
with open('styles.html') as f:
    custom_css = f.read()

# Apply the custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

col1, col2, col3 = st.columns([12, 12, 2], vertical_alignment='center')
with col1:
    st.image('logo.png', width=400)

with col2:
    st.image('image.png', width=200)

with col3:
    st.image('WJ4.png', width=100)

################################# MACHINE LEARNING ########################################
is_csv = False
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    uploaded_data = st.file_uploader("Masukkan data", type=['csv', 'xlsx'])

def load_data(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file).reset_index()
    elif file.name.endswith('.xlsx') or file.name.endswith('.xls'):
        df = pd.read_excel(file, engine='openpyxl').reset_index()
    else:
        raise ValueError("File type not supported. Please upload a CSV or Excel file.")
    return df

if uploaded_data is None:
    st.info("Masukkan data yang akan dicek")
    st.stop()

df = load_data(uploaded_data)

data = df.copy()

# Anomaly detection based on KWH and KWH_pred with tolerance of 206
kwh_pred = pickle.load(open('KWH_xgbrf_model.pkl', 'rb'))

cols_kwh = ['Daya Master', 'RRC Average', 'User Active Average', 'Payload', 'Total RRU', 'KWH']
pred1_df = df[cols_kwh]
X1_pred_df = pred1_df.drop(columns='KWH', axis=1)
pred1_df['KWH_pred'] = kwh_pred.predict(X1_pred_df)
pred1_df['KWH_pred'] = pred1_df['KWH_pred'].astype(int)

df['KWH_pred'] = pred1_df['KWH_pred']

# Check anomaly based on KWH and KWH_pred
def anomaly_check(data):
    tolerance = 206  # KWH tolerance value
    data['anomaly'] = data.apply(lambda row: 'Anomali' if abs(row['KWH'] - row['KWH_pred']) > tolerance else 'Normal', axis=1)

anomaly_check(df)

# Merge additional columns and handle missing columns as needed
tambah = pd.read_csv('dataset_train/tambahan.csv')
tambah['Kota / Kabupaten'] = tambah['Kota / Kabupaten'].replace('NOP ', '', regex=True)
tambah = tambah.drop(columns=['Site Name', 'Area', 'Regional'], axis=1)

df = df.merge(tambah, on='Site ID', how='left')
df = df.dropna()

if 'Cost' not in df.columns:
    df['Cost'] = 0

cols = df.columns.tolist()
cols.insert(3, cols.pop(cols.index('Kota / Kabupaten')))
df = df.reindex(columns=cols)

# Calculate Cost_pred before determine_keterangan
cost_pred_cols = ['Daya Master', 'RRC Average', 'User Active Average', 'Payload', 'Total RRU']
pred2_df = df[cost_pred_cols]
X2_pred_df = pred2_df  # Add any necessary feature engineering or transformations
pred2_df['Cost_pred'] = model_cost.predict(X2_pred_df)

df['Cost_pred'] = pred2_df['Cost_pred']

# Modify the 'determine_keterangan' function to compare 'Cost' and 'Cost_pred'
def determine_keterangan(row):
    tolerance = 355630  # Cost tolerance value
    if abs(row['Cost'] - row['Cost_pred']) > tolerance:
        return 'Over Baseline'
    elif row['Cost'] < row['Cost_pred']:
        return 'Below Baseline'
    else:
        return 'In Baseline'

df['keterangan'] = df.apply(determine_keterangan, axis=1)

################################# DISPLAY FUNCTION ########################################

# Split layout into two columns
col1, col2 = st.columns([1.5, 1])

with col1:

    col_controls1, col_controls2 = st.columns([1, 1])
    with col_controls1:
        selected_city = st.selectbox('Pilih Kota/Kabupaten:', ['Semua'] + sorted(df['Kota / Kabupaten'].unique().tolist()))
    with col_controls2:
        selected_keterangan = st.selectbox('Pilih Keterangan:', ['Semua'] + df['keterangan'].unique().tolist())

    filtered_df = df
    city_df = df
    if selected_city != 'Semua':
        filtered_df = filtered_df[filtered_df['Kota / Kabupaten'] == selected_city]
        city_df = city_df[city_df['Kota / Kabupaten'] == selected_city]
    if selected_keterangan != 'Semua':
        filtered_df = filtered_df[filtered_df['keterangan'] == selected_keterangan]

    scatter_plot = alt.Chart(filtered_df).mark_circle(size=60).encode(
        x='index',
        y='KWH',
        color=alt.condition(
            alt.datum.anomaly == 'Anomali',
            alt.value('red'),
            alt.value('blue')
        ),
        tooltip=[
            alt.Tooltip('Site ID', title='Site ID'),
            alt.Tooltip('Site Name', title='Site Name'),
            alt.Tooltip('KWH', title='KWH')
        ]
    ).properties(
        width=800,
        height=400,
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    ).configure_title(
        fontSize=16
    )

    st.altair_chart(scatter_plot, use_container_width=True)


with col2:
    in_cols1, in_cols2 = st.columns([1, 1])

    with in_cols1: 
        total_cost = city_df['Cost'].sum()
        formatted_cost = f"Rp. {total_cost:,.0f}".replace(',', '.')
        st.metric("Total Pembayaran", formatted_cost)

        ano = city_df[city_df['anomaly'] == 'Anomali'].shape[0]
        formatted_anomaly = f"{ano} / {city_df.shape[0]} site"
        st.metric("Jumlah Anomali", formatted_anomaly)

    with in_cols2:
        rugi_df = city_df[city_df['keterangan'] == 'Over Baseline'] 
        cols = ['Daya Master', 'RRC Average', 'User Active Average', 'Payload', 'Total RRU']
        rugi_dataset = rugi_df[cols]
        rugi_df['cost_asli'] = model_cost.predict(rugi_dataset)
        rugi2_cost = rugi_df['cost_asli'].sum()
        formatted_rugi2_cost = f"Rp. {rugi2_cost:,.0f}".replace(',', '.')
        st.metric("Perkiraan total Anomali", formatted_rugi2_cost)

        rugi_cost = rugi_df['Cost'].sum() - rugi_df["cost_asli"].sum()
        formatted_rugi_cost = f"- Rp. {rugi_cost:,.0f}".replace(',', '.')
        st.metric("Perkiraan total Kerugian Anomali", formatted_rugi_cost)

    untung_cost = total_cost - rugi_cost
    formatted_untung_cost = f"Rp. {untung_cost:,.0f}".replace(',', '.')
    st.metric("Total Pembayaran Tanpa Anomali", formatted_untung_cost)        

    OB, IB, BB = st.columns([1, 1, 1])

    with OB:    
        OB = city_df[city_df['keterangan'] == 'Over Baseline'].shape[0]
        OB_f = f"{OB} site"
        st.metric("Over Baseline", OB_f)
    
    with IB:
        IB = city_df[city_df['keterangan'] == 'In Baseline'].shape[0]
        IB_f = f"{IB} site"
        st.metric("In Baseline", IB_f)

    with BB:
        BB = city_df[city_df['keterangan'] == 'Below Baseline'].shape[0]
        BB_f = f"{BB} site"
        st.metric("Below Baseline", BB_f)

# Display the predictions for KWH and Cost
df['KWH_pred'] = pred1_df['KWH_pred']

# Handle cost prediction
cost_pred_cols = ['Daya Master', 'RRC Average', 'User Active Average', 'Payload', 'Total RRU']
pred2_df = df[cost_pred_cols]
X2_pred_df = pred2_df  # Add any necessary feature engineering or transformations
pred2_df['Cost_pred'] = model_cost.predict(X2_pred_df)

df['Cost_pred'] = pred2_df['Cost_pred']

# Display the dataframe
st.dataframe(filtered_df.drop(columns=['index'], axis=1))

col1, col2 = st.columns([1.4 , 0.6]) 
with col1:
    st.text('* Prediksi Harga bisa melenceng ± Rp 355.630')
    st.text('* Prediksi KWH bisa melenceng ± 206 KWH')

with col2:
    select, button = st.columns([0.7 , 1.3 ], vertical_alignment='bottom')

    def to_excel(dataframe):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            dataframe.to_excel(writer, index=False, sheet_name='Sheet1')
        processed_data = output.getvalue()
        return processed_data

    with select:
        file_format = st.selectbox("Pilih format", ["CSV", "Excel"])
        if file_format == "CSV":
            csv_data = filtered_df.to_csv(index=False)
            download_data = csv_data
            mime_type = "text/csv"
            file_extension = "csv"
        else:
            excel_data = to_excel(filtered_df)
            download_data = excel_data
            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            file_extension = "xlsx"

    with button:
        st.download_button(
            label=f"Download data sebagai {file_format}",
            data=download_data,
            file_name=f"Data Anomali_{selected_city}_{selected_keterangan}_Bulan {df['Month'].iloc[1]}.{file_extension}",
            mime=mime_type
        )
