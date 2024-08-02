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

st.title("Analisis Tegangan Listrik Anomali Singkat")
st.sidebar.image('logo_white.png')
with open('styles.html') as f:
    custom_css = f.read()

# Apply the custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

################################# MACHINE LEARNING ########################################
is_csv = False
def load_data(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file).reset_index()
        is_csv = True
    elif file.name.endswith('.xlsx') or file.name.endswith('.xls'):
        df = pd.read_excel(file, engine='openpyxl').reset_index()
        is_csv = False
    else:
        raise ValueError("File type not supported. Please upload a CSV or Excel file.")
    return df

uploaded_data = st.sidebar.file_uploader("Masukkan data", type=['csv', 'xlsx'])

if uploaded_data is None:
    st.sidebar.info("Masukkan data yang akan dicek")
    st.stop()

df = load_data(uploaded_data)

data = df.copy()
columns_to_drop = ['Site ID', 'Site Name', 'Month', 'Site Tipe']
data = data.drop(columns=columns_to_drop, errors='ignore')

def anomaly_check(data):
    data['anomaly'] = pipe.predict(data[["KWH"]])

anomaly_check(data)

data['anomaly'] = data['anomaly'].apply(lambda x: 'Normal' if x == 1 else 'Anomali')
df['anomaly'] = data['anomaly']

anomali = df[df['anomaly'] == 'Anomali']

col = ['Site Name', 'Area', 'Regional']

tambah = pd.read_csv('dataset_train/tambahan.csv')
tambah['Kota / Kabupaten'] = tambah['Kota / Kabupaten'].replace('NOP ', '', regex=True)
tambah = tambah.drop(columns=col, axis=1)

df = df.merge(tambah, on='Site ID', how='left')

cols = df.columns.tolist()
cols.insert(3, cols.pop(cols.index('Kota / Kabupaten')))
df = df.reindex(columns=cols)

def determine_keterangan(row):
    if row['KWH'] > row['Emax (kWh)']:
        return 'Over Baseline'
    elif row['KWH'] < row['Emin40 (kWh)']:
        return 'Below Baseline'
    else:
        return 'In Baseline'

df['keterangan'] = df.apply(determine_keterangan, axis=1)

################################# DISPLAY FUNCTION ########################################

def display(df):
    # Filter data based on selected values

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
            width= 800,
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

        with in_cols1 : 
            total_cost = city_df['Cost'].sum()
            formatted_cost = f"Rp. {total_cost:,.0f}".replace(',', '.')
            st.metric("Total Pembayaran", formatted_cost)

            ano = city_df[city_df['anomaly'] == 'Anomali'].shape[0]
            formatted_anomaly = f"{ano} / {city_df.shape[0]} site"
            st.metric("Jumlah Anomali", formatted_anomaly)

        with in_cols2 :
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

    kwh_pred = pickle.load(open('KWH_xgbrf_model.pkl', 'rb'))
    
    cols_kwh = ['Daya Master','RRC Average','User Active Average','Payload','Total RRU', 'KWH']
    pred1_df = filtered_df[filtered_df['keterangan'] == 'Over Baseline']
    pred1_df = pred1_df[cols_kwh]
    X1_pred_df = pred1_df.drop(columns='KWH', axis=1)
    pred1_df['KWH_pred'] = kwh_pred.predict(X1_pred_df)
    pred1_df['KWH_pred'] = pred1_df['KWH_pred'].astype(int)

    cols_cost = ['Daya Master','RRC Average','User Active Average','Payload','Total RRU', 'Cost']
    pred2_df = filtered_df[filtered_df['keterangan'] == 'Over Baseline']
    pred2_df = pred2_df[cols_cost]
    X2_pred_df = pred2_df.drop(columns='Cost', axis=1)
    pred2_df['Cost_pred'] = model_cost.predict(X2_pred_df)
    pred2_df['Cost_pred'] = pred2_df['Cost_pred'].astype(int)

    filtered_df['KWH_pred'] = pred1_df['KWH_pred']
    filtered_df['Cost_pred'] = pred2_df['Cost_pred']

    st.dataframe(filtered_df.drop(columns=['index'], axis=1))
    col1, col2 = st.columns([1.4 , 0.6]) 
    with col1 :
        st.text('* Prediksi Harga bisa melenceng ± Rp 355.630')
        st.text('* Prediksi KWH bisa melenceng ± 206 KWH')
    with col2 :

        select, button = st.columns([0.7 , 1.3 ] , vertical_alignment = 'bottom')
        
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
                excel_data = to_excel(df)
                download_data = excel_data
                mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                file_extension = "xlsx"
        
        with button:
            st.download_button(
                label=f"Download data sebagai {file_format}",
                data=download_data,
                file_name=f"data.{file_extension}",
                mime=mime_type
            )

    
################################# UI ########################################

# Call display function with filters
display(df)
