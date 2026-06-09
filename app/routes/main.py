import json, os, pandas as pd, plotly, plotly.express as px, pycountry
import plotly.graph_objects as go
from flask import Blueprint, render_template, jsonify
from app import cache

main = Blueprint('main', __name__)
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'processed', 'processed_faers.csv')
KOREA_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'raw', 'korea_adr.csv')


def load_df():
    return pd.read_csv(DATA_PATH)

def alpha2_to_alpha3(code):
    try:
        return pycountry.countries.get(alpha_2=code).alpha_3
    except:
        return None
    
@main.route('/')
def index():
    return render_template('index.html')


@main.route('/dashboard')
@cache.cached(timeout=300)
def dashboard():
    df = load_df()
    top_reac = df['pt'].value_counts().head(20).reset_index()
    top_reac.columns = ['reaction', 'count']
    fig1 = px.bar(top_reac, x='reaction', y='count', title='Top 20 Adverse Reactions', color='count', color_continuous_scale='Blues')
    fig1.update_layout(xaxis_tickangle=-45, template='plotly_dark', height=420)
    top_drug = df['drugname'].value_counts().head(20).reset_index()
    top_drug.columns = ['drug', 'count']
    fig2 = px.bar(top_drug, x='drug', y='count', title='Top 20 Drugs by Report Count', color='count', color_continuous_scale='Reds')
    fig2.update_layout(xaxis_tickangle=-45, template='plotly_dark', height=420)
    sex_counts = df['sex'].value_counts().reset_index()
    sex_counts.columns = ['sex', 'count']
    fig3 = px.pie(sex_counts, names='sex', values='count', title='Gender Distribution', template='plotly_dark')
    fig3.update_layout(height=420)
    outc_counts = df['outc_cod'].value_counts().reset_index()
    outc_counts.columns = ['outcome', 'count']
    fig4 = px.bar(outc_counts, x='outcome', y='count', title='Outcome Distribution', color='count', color_continuous_scale='Greens')
    fig4.update_layout(template='plotly_dark', height=420)
    df['age_group'] = pd.cut(df['age'], bins=[0,18,30,45,60,75,120], labels=['0-18','19-30','31-45','46-60','61-75','76+'])
    age_counts = df['age_group'].value_counts().sort_index().reset_index()
    age_counts.columns = ['age_group', 'count']
    fig5 = px.bar(age_counts, x='age_group', y='count', title='Age Group Report Count', color='count', color_continuous_scale='Purples')
    fig5.update_layout(template='plotly_dark', height=420)
    country_counts = df['reporter_country'].value_counts().reset_index()
    country_counts.columns = ['country', 'count']
    country_counts['iso3'] = country_counts['country'].apply(alpha2_to_alpha3)
    country_counts = country_counts.dropna(subset=['iso3'])
    fig6 = px.scatter_geo(country_counts, locations='iso3', size='count', hover_name='country', title='Report Count by Country', color='count', color_continuous_scale='Reds', projection='natural earth')
    fig6.update_layout(template='plotly_dark', height=420, geo=dict(showframe=False, showcoastlines=True, showland=True, landcolor='#1e293b', bgcolor='#0f172a'))
    charts = {'chart1': json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder), 'chart2': json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder), 'chart3': json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder), 'chart4': json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder), 'chart5': json.dumps(fig5, cls=plotly.utils.PlotlyJSONEncoder), 'chart6': json.dumps(fig6, cls=plotly.utils.PlotlyJSONEncoder)}
    return render_template('dashboard.html', charts=charts)


@main.route('/korea')
def korea_dashboard():
    df = pd.read_csv(KOREA_DATA_PATH, encoding='utf-8')
    df.columns = ['rank','sym_2024','cnt_2024','sym_2023','cnt_2023','sym_2022','cnt_2022','sym_2021','cnt_2021','sym_2020','cnt_2020','sym_2019','cnt_2019']
    fig1 = px.bar(df.head(10), x='sym_2024', y='cnt_2024', title='Korea 2024 Top 10 ADR', color='cnt_2024', color_continuous_scale='Blues')
    fig1.update_layout(xaxis_tickangle=-45, template='plotly_dark', height=420)
    years = ['2019','2020','2021','2022','2023','2024']
    top5 = df.head(5)['sym_2024'].tolist()
    fig2 = px.line(title='Korea Top 5 ADR Trend')
    for symptom in top5:
        counts = []
        for y in years:
            col = f'cnt_{y}'
            row = df[df['sym_2024'] == symptom]
            counts.append(int(row[col].values[0]) if len(row) > 0 and col in df.columns else 0)
        fig2.add_scatter(x=years, y=counts, name=symptom, mode='lines+markers')
    fig2.update_layout(template='plotly_dark', height=420)

    df10 = df.head(10)
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=df10['sym_2024'], y=df10['cnt_2024'], name='2024', marker_color='#38bdf8'))
    fig3.add_trace(go.Bar(x=df10['sym_2024'], y=df10['cnt_2023'], name='2023', marker_color='#a78bfa'))
    fig3.update_layout(barmode='group', title='2024 vs 2023 Top 10 ADR', xaxis_tickangle=-45, template='plotly_dark', height=420)

    fig3.update_layout(xaxis_tickangle=-45, template='plotly_dark', height=420)
    charts = {'chart1': json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder), 'chart2': json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder), 'chart3': json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)}
    return render_template('korea.html', charts=charts)
