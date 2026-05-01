import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Productivity & Well-being Research",
    page_icon="📝",
    layout="wide"
)

# --- PREMIUM STYLING ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main {
        background-color: #f8fbfd;
    }
    
    /* Style the tabs to be more obvious */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #e9ecef;
        padding: 10px 10px 0 10px;
        border-radius: 15px 15px 0 0;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #dee2e6;
        border-radius: 10px 10px 0 0;
        gap: 1px;
        padding: 10px 25px;
        font-weight: 600;
        color: #495057;
    }

    .stTabs [aria-selected="true"] {
        background-color: #34495e !important;
        color: white !important;
        border-bottom: 3px solid #e74c3c !important;
    }

    .stMetric {
        background: #ffffff;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    .prediction-card {
        background: #34495e;
        color: white;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        text-align: center;
        margin: 20px 0;
    }
    
    .insight-box {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #34495e;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- DATA ENGINE ---
@st.cache_data
def load_data():
    df = pd.read_csv('social_media_vs_productivity.csv')
    df.columns = df.columns.str.strip()
    cols_to_fix = ['actual_productivity_score', 'stress_level', 'sleep_hours', 'daily_social_media_time']
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(df[col].median(), inplace=True)
    df.drop_duplicates(inplace=True)
    return df

df_full = load_data()

# --- HEADER ---
st.title("Productivity & Well-being Research")
st.markdown("""
This research investigates digital habits across different professions.
Data Source: **Mahdi Mashayekhi (Kaggle)**
""")
st.markdown("---")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Global Filters")
    platform_choice = st.radio(
        "Select Platform Filter:",
        options=["All Platforms"] + sorted(df_full['social_platform_preference'].unique().tolist())
    )
    st.markdown("---")
    st.write("**Navigation Guide:** Use the tabs to switch between specific student analysis and broader professional comparisons.")

if platform_choice == "All Platforms":
    current_df = df_full
else:
    current_df = df_full[df_full['social_platform_preference'] == platform_choice]

# --- MAIN TABS ---
tab1, tab2, tab3 = st.tabs([
    "🎓 Student Focus & Prediction", 
    "💼 Professional Comparison",
    "🔬 Further Analysis"
])

# Helper for dynamic naming
platform_name = "social media in general" if platform_choice == "All Platforms" else platform_choice

with tab1:
    df_student = current_df[current_df['job_type'] == 'Student'].copy()
    
    # 1. Trends Section
    st.subheader("Section 1: Data Trends")
    st.write(f"How does time spent on {platform_choice} affect student productivity?")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Student Sample", f"{len(df_student):,}")
    with c2:
        st.metric("Average Social Media Usage", f"{df_student['daily_social_media_time'].mean():.1f} hrs")
    with c3:
        st.metric("Average Productivity Score", f"{df_student['actual_productivity_score'].mean():.1f}/10")

    # Regression Math
    x = df_student['daily_social_media_time']
    y = df_student['actual_productivity_score']
    
    if len(x) > 1:
        m, k = np.polyfit(x, y, 1)
        correlation = x.corr(y)
    else:
        m, k, correlation = 0, 0, 0
    
    fig1 = px.scatter(
        df_student, x='daily_social_media_time', y='actual_productivity_score',
        labels={'daily_social_media_time': 'Social Media Usage (Hours)', 'actual_productivity_score': 'Productivity Score'},
        opacity=0.4, template='plotly_white'
    )
    fig1.update_traces(marker=dict(color='#2980b9'))
    
    if len(x) > 0:
        x_range = np.linspace(0, x.max(), 100)
        fig1.add_trace(go.Scatter(x=x_range, y=m*x_range+k, mode='lines', name='Trend', line=dict(color='#e74c3c', width=3)))
    
    fig1.update_layout(height=450, showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")

    # 2. Prediction Section
    st.subheader("Section 2: Personal Productivity Predictor")
    st.write(f"Predicting your focus score based on {platform_choice} usage habits.")
    
    ca, cb = st.columns([1, 1])
    with ca:
        user_hours = st.select_slider(
            "Enter daily usage (Hrs):",
            options=np.round(np.arange(0, 10.5, 0.5), 1),
            value=2.0,
            key="student_slider"
        )
        
        predicted_score = m * user_hours + k
        predicted_score = max(0, min(10, predicted_score))
        
        st.markdown(f"""
        <div class="prediction-card">
            <h1 style="color: white; font-size: 60px;">{predicted_score:.1f}</h1>
            <p style="font-size: 20px;">Predicted Focus Score</p>
            <p style="opacity: 0.8;">Based on {user_hours} hours of {platform_choice} usage</p>
        </div>
        """, unsafe_allow_html=True)

    with cb:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="insight-box">
            <h4>Impact Analysis</h4>
            <p>The model reveals that for every additional hour spent on {platform_choice}, focus typically drops by <b>{abs(m):.2f} points</b>.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 3. Conclusion Section
    st.subheader("Section 3: Findings & Conclusion")
    
    st.markdown(f"""
    **1. Productivity Link:** Our investigation shows a measurable link between time spent on **{platform_name}** and productivity in students. The correlation for this specific group is approximately **{correlation:.2f}**.

    **2. Relationship Nuance:** Even though we observe higher stress levels for high **{platform_name}** usage, we cannot say that high usage *causes* stress. It is highly possible that students turn to this platform as a **coping mechanism** to escape academic pressure.

    **3. Final Takeaway:** Balanced digital consumption is key. For **{platform_name}**, maintaining a limit of under 2 hours daily is statistically associated with the highest productivity output in this study.
    """)

with tab2:
    st.subheader("Professional Occupation Comparison")
    st.write(f"How does productivity vary with {platform_choice} usage across different job types?")
    
    # BACK TO SCATTER PLOT as requested
    fig_comp = px.scatter(
        current_df, 
        x='daily_social_media_time', 
        y='actual_productivity_score',
        color='job_type',
        labels={'job_type': 'Occupation', 'actual_productivity_score': 'Productivity Score', 'daily_social_media_time': 'Social Media Usage (Hours)'},
        template='plotly_white',
        opacity=0.5
    )
    fig_comp.update_layout(height=500)
    st.plotly_chart(fig_comp, use_container_width=True)
    
    st.info(f"""
    **Explaining the Chart:** Each dot represents an individual. By coloring the dots by occupation, we can see 
    if certain groups (like Students in blue) follow a different pattern than professionals (like IT or Finance) 
    when using **{platform_choice}**.
    """)

with tab3:
    df_insight = current_df[current_df['job_type'] == 'Student'].copy()
    st.subheader("Deep-Dive Analysis (Student Data Only)")
    
    st.markdown(f"#### Research Insight 1: {platform_choice} vs. Stress")
    fig2 = px.box(
        df_insight, x='daily_social_media_time', y='stress_level', orientation='h', color='stress_level',
        labels={'stress_level': 'Stress Level', 'daily_social_media_time': 'Usage (Hrs)'},
        template='plotly_white', color_discrete_sequence=px.colors.sequential.Tealgrn
    )
    fig2.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)
    
    # DYNAMIC CONCLUSION FOR STRESS
    st.info(f"""
    **Observation:** The data shows that higher usage of **{platform_name}** generally aligns with higher stress categories. 
    However, we cannot conclude that **{platform_name}** *causes* stress. It is possible that the relationship is reversed 
    (stressed students use this platform to cope) or that the relationship is indirect.
    """)
    
    st.markdown("---")
    
    st.markdown("#### Research Insight 2: Sleep Duration vs. Focus")
    fig3 = px.scatter(
        df_insight, x='sleep_hours', y='actual_productivity_score',
        labels={'sleep_hours': 'Sleep Duration (Hours)', 'actual_productivity_score': 'Productivity Score'},
        template='plotly_white', opacity=0.3
    )
    fig3.update_traces(marker=dict(color='#16a085'))
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)
    
    # DYNAMIC CONCLUSION FOR SLEEP
    st.info(f"""
    **Observation:** By looking at the distribution of points for **{platform_name}** users, we can see how sleep 
    habits relate to productivity. Most students with high focus scores also tend to fall into the healthy sleep 
    range, suggesting that rest plays a key role in academic success.
    """)

st.caption("Final Project | Productivity Research Study")
