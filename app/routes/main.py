import json
import os
import pandas as pd
import plotly
import plotly.express as px
import pycountry
 
from flask import Blueprint, render_template, jsonify
from app import cache
 
main = Blueprint('main', __name__)
 
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         'data', 'processed', 'processed_faers.csv')
KOREA_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                               'data', 'raw', 'korea_adr.csv')
 
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
    fig1 = px.bar(top_reac, x='reaction', y='count',
                  title='Top 20 Adverse Reactions',
                  color='count', color_continuous_scale='Blues')
    fig1.update_layout(xaxis_tickangle=-45, template='plotly_dark', height=420)
 
    top_drug = df['drugname'].value_counts().head(20).reset_index()
    top_drug.columns = ['drug', 'count']
    fig2 = px.bar(top_drug, x='drug', y='count',
                  title='Top 20 Drugs by Report Count',
                  color='count', color_continuous_scale='Reds')
    fig2.update_layout(xaxis_tickangle=-45, template='plotly_dark', height=420)
 
    sex_counts = df['sex'].value_counts().reset_index()
    sex_counts.columns = ['sex', 'count']
    fig3 = px.pie(sex_counts, names='sex', values='count',
                  title='Gender Distribution',
                  template='plotly_dark')
    fig3.update_layout(height=420)
 
    outc_counts = df['outc_cod'].value_counts().reset_index()
    outc_counts.columns = ['outcome', 'count']
    fig4 = px.bar(outc_counts, x='outcome', y='count',
                  title='Outcome Distribution',
                  color='count', color_continuous_scale='Greens')
    fig4.update_layout(template='plotly_dark', height=420)
 
    df['age_group'] = pd.cut(df['age'],
        bins=[0, 18, 30, 45, 60, 75, 120],
        labels=['0-18', '19-30', '31-45', '46-60', '61-75', '76+']
    )
    age_counts = df['age_group'].value_counts().sort_index().reset_index()
    age_counts.columns = ['age_group', 'count']
    fig5 = px.bar(age_counts, x='age_group', y='count',
                  title='?°ë ¹ë³?ë³´ê³  ê±´ìˆ˜',
                  color='count', color_continuous_scale='Purples')
    fig5.update_layout(template='plotly_dark', height=420)
 
    country_counts = df['reporter_country'].value_counts().reset_index()
    country_counts.columns = ['country', 'count']
    country_counts['iso3'] = country_counts['country'].apply(alpha2_to_alpha3)
    country_counts = country_counts.dropna(subset=['iso3'])
    fig6 = px.scatter_geo(
        country_counts, locations='iso3', size='count',
        hover_name='country', title='êµ??ë³?ë³´ê³  ê±´ìˆ˜',
        color='count', color_continuous_scale='Reds', projection='natural earth'
    )
    fig6.update_layout(template='plotly_dark', height=420,
        geo=dict(showframe=False, showcoastlines=True, showland=True,
                 landcolor='#1e293b', bgcolor='#0f172a'))
 
    charts = {
        'chart1': json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder),
        'chart2': json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder),
        'chart3': json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder),
        'chart4': json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder),
        'chart5': json.dumps(fig5, cls=plotly.utils.PlotlyJSONEncoder),
        'chart6': json.dumps(fig6, cls=plotly.utils.PlotlyJSONEncoder),
    }
    return render_template('dashboard.html', charts=charts)
 
@main.route('/korea')
def korea_dashboard():
    # 1. ?¸ì½”??? ê²½ ?°ì? ?Šê³  ?ˆì „?˜ê²Œ ?Œì¼ ?½ê¸°
    df = pd.read_csv(KOREA_DATA_PATH)
    
    # 2. ??[?µì‹¬] ?œê? ë§¤ì¹­ ?¤íŒ¨ë¥?ë°©ì??˜ê¸° ?„í•´ ì»¬ëŸ¼ '?œì„œ'?€ë¡??´ë¦„??ê°•ì œ ì£¼ì… ??
    # ?Œì¼??ì¡´ì¬?˜ëŠ” ì»¬ëŸ¼ ?˜ì— ë§ì¶° ?ì—?œë???ê°•ì œë¡???–´?Œì›?ˆë‹¤.
    new_columns = [
        'rank',       # 1ë²ˆì§¸: ?œìœ„
        'sym_2024',   # 2ë²ˆì§¸: ?°ë„ë³„ì¦??2024)
        'cnt_2024',   # 3ë²ˆì§¸: ?°ë„ë³„ë³´ê³ ê±´??2024)
        'sym_2023',   # 4ë²ˆì§¸: ?°ë„ë³„ì¦??2023)
        'cnt_2023',   # 5ë²ˆì§¸: ?°ë„ë³„ë³´ê³ ê±´??2023)
        'sym_2022',   # 6ë²ˆì§¸: ?°ë„ë³„ì¦??2022)
        'cnt_2022',   # 7ë²ˆì§¸: ?°ë„ë³„ë³´ê³ ê±´??2022)
        'sym_2021',   # 8ë²ˆì§¸: ?°ë„ë³„ì¦??2021)
        'cnt_2021',   # 9ë²ˆì§¸: ?°ë„ë³„ë³´ê³ ê±´??2021)
        'sym_2020',   # 10ë²ˆì§¸: ?°ë„ë³„ì¦??2020)
        'cnt_2020',   # 11ë²ˆì§¸: ?°ë„ë³„ë³´ê³ ê±´??2020)
        'sym_2019',   # 12ë²ˆì§¸: ?°ë„ë³„ì¦??2019)
        'cnt_2019'    # 13ë²ˆì§¸: ?°ë„ë³„ë³´ê³ ê±´??2019)
    ]
    
    # ë§Œì•½ ?Œì¼???¤ì œ ì»¬ëŸ¼ ?˜ì? ?°ë¦¬ê°€ ì§€?•í•œ ê°œìˆ˜ê°€ ?¤ë? ê²½ìš°ë¥??€ë¹„í•œ ?ˆì „ ?„í„°ë§?
    df.columns = new_columns[:len(df.columns)]

    # 3. ë³€??ì§€??
    sym_col  = 'sym_2024'
    cnt_2024 = 'cnt_2024'
    cnt_2023 = 'cnt_2023'

    # ì²?ë²ˆì§¸ ê·¸ë˜??(2024 Top 10)
    fig1 = px.bar(df.head(10), x=sym_col, y=cnt_2024,
                  title='?œêµ­ 2024??Top 10 ?´ìƒë°˜ì‘',
                  color=cnt_2024, color_continuous_scale='Blues')
    fig1.update_layout(xaxis_tickangle=-45, template='plotly_dark', height=420)

    # ??ë²ˆì§¸ ê·¸ë˜??(Top 5 ?°ë„ë³?ì¶”ì´)
    years = ['2019', '2020', '2021', '2022', '2023', '2024']
    top5 = df.head(5)[sym_col].tolist()
    fig2 = px.line(title='?œêµ­ Top 5 ?´ìƒë°˜ì‘ ?°ë„ë³?ì¶”ì´')
    for symptom in top5:
        counts = []
        for y in years:
            col = f'cnt_{y}'
            row = df[df[sym_col] == symptom]
            # ì»¬ëŸ¼???¨ì „??ì¡´ì¬???Œë§Œ ?°ì´?°ë? ?•ìˆ˜?•ìœ¼ë¡?ì¶”ì¶œ
            if len(row) > 0 and col in df.columns:
                counts.append(int(row[col].values[0]))
            else:
                counts.append(0)
        fig2.add_scatter(x=years, y=counts, name=symptom, mode='lines+markers')
    fig2.update_layout(template='plotly_dark', height=420)

    # ??ë²ˆì§¸ ê·¸ë˜??(2024 vs 2023 ë¹„êµ)
    fig3 = px.bar(df.head(10), x=sym_col, y=['¿¬µµº°º¸°í°Ç¼ö(2024)', '¿¬µµº°º¸°í°Ç¼ö(2023)'],
                  title='2024 vs 2023 Top 10 ?´ìƒë°˜ì‘ ë¹„êµ',
                  barmode='group', color_discrete_sequence=['#38bdf8', '#a78bfa'])
    fig3.update_layout(xaxis_tickangle=-45, template='plotly_dark', height=420)

    charts = {
        'chart1': json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder),
        'chart2': json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder),
        'chart3': json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder),
    }
    return render_template('korea.html', charts=charts)
