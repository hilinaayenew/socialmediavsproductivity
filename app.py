import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Productivity Insight",
    page_icon="🚀",
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
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .stMetric {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        padding: 25px;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        transition: transform 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-5px);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 25px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
        text-align: center;
        margin: 20px 0;
    }
    
    .sidebar .sidebar-content {
        background-color: #ffffff;
    }
    
    .insight-box {
        background: white;
        padding: 20px;
        border-radius: 15px;
        border-left: 6px solid #667eea;
        margin-bottom: 20px;
    }
    
    .stButton>button {
        background: linear-gradient(to right, #667eea, #764ba2);
        color: white;
        border-radius: 30px;
        border: none;
        padding: 10px 25px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        transform: scale(1.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- DATA ENGINE ---
@st.cache_data
def load_data():
    df = pd.read_csv('social_media_vs_productivity.csv')
    df = df[df['job_type'] == 'Student'].copy()
    for col in ['actual_productivity_score', 'stress_level', 'sleep_hours']:
        df[col].fillna(df[col].median(), inplace=True)
    df.dropna(subset=['daily_social_media_time', 'actual_productivity_score'], inplace=True)
    df.drop_duplicates(inplace=True)
    return df

df = load_data()

# --- HEADER SECTION ---
with st.container():
    st.title("🚀 Student Productivity Insight")
    st.markdown("""
    Use this interactive tool to explore how social media habits impact academic performance. 
    **How to use:** Start by selecting a platform in the sidebar, then browse the tabs to see data trends, 
    predict your own score, and read our final conclusions.
    """)
    st.markdown("---")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Platform Analysis")
    platform_choice = st.radio(
        "Select a Social Platform:",
        options=["All Platforms"] + sorted(df['social_platform_preference'].unique().tolist())
    )
    st.markdown("---")
    st.info("💡 **Guide:** Selecting a specific platform will update all the charts and the 'Hour Cost' calculation to show data only for students who prefer that app.")

if platform_choice == "All Platforms":
    filtered_df = df
else:
    filtered_df = df[df['social_platform_preference'] == platform_choice]

# --- MAIN DASHBOARD ---
tab1, tab2, tab3 = st.tabs(["📊 Data Trends", "🔮 Predictor", "📍 Conclusion"])

with tab1:
    st.markdown("### Step 1: Observe the Trends")
    st.write("First, let's look at the raw data. Each dot represents a student, and the red line shows the average trend.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Sample", f"{len(filtered_df):,}", "Students")
    with col2:
        st.metric("Avg. Usage", f"{filtered_df['daily_social_media_time'].mean():.1f} hrs", "Daily")
    with col3:
        st.metric("Focus Score", f"{filtered_df['actual_productivity_score'].mean():.1f}/10", "Avg.")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # INTERACTIVE PLOTLY CHART
    st.subheader("The Productivity Spectrum")
    
    # Regression Math
    x = filtered_df['daily_social_media_time']
    y = filtered_df['actual_productivity_score']
    m, k = np.polyfit(x, y, 1)
    
    fig = px.scatter(
        filtered_df, 
        x='daily_social_media_time', 
        y='actual_productivity_score',
        hover_data=['social_platform_preference', 'sleep_hours', 'stress_level'],
        labels={'daily_social_media_time': 'Usage (Hrs)', 'actual_productivity_score': 'Productivity'},
        opacity=0.4,
        template='plotly_white'
    )
    # Set uniform color for points
    fig.update_traces(marker=dict(color='#3498db'))
    
    # Add Trend Line
    x_range = np.linspace(0, x.max(), 100)
    y_range = m * x_range + k
    fig.add_trace(go.Scatter(
        x=x_range, 
        y=y_range, 
        mode='lines', 
        name='Trend (Best Fit Line)', 
        line=dict(color='red', width=3)
    ))
    
    fig.update_layout(
        height=500, 
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### Step 2: Test the Model")
    st.write("Now that we've seen the data, let's see where you might fit into the pattern.")
    
    c1, c2 = st.columns([1, 1])
    with c1:
        user_hours = st.select_slider(
            "Slide to your average social media usage (Hrs/Day):",
            options=np.round(np.arange(0, 10.5, 0.5), 1),
            value=2.0
        )
        
        predicted_score = m * user_hours + k
        predicted_score = max(0, min(10, predicted_score))
        
        st.markdown(f"""
        <div class="prediction-card">
            <h1 style="color: white; font-size: 60px;">{predicted_score:.1f}</h1>
            <p style="font-size: 20px;">Predicted Productivity Score</p>
            <p style="opacity: 0.8;">Based on your {user_hours}h usage profile</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="insight-box">
            <h4>The "Hour Cost"</h4>
            <p>The model shows that every additional hour of social media is linked to a <b>{abs(m):.2f} point</b> drop in focus.</p>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.markdown("### Step 3: Read the Conclusion")
    st.write("Finally, we summarize the relationship between usage and focus.")
    st.subheader("Final Summary & Findings")
    st.markdown(f"""
    ### 1. The Numbers Don't Lie
    Based on this specific data, there is a clear link: **more social media time = lower productivity.** 
    - The model shows a 'Pearson Correlation' of about **-0.62**. In simple terms, this means that as one goes up, the other almost always goes down.
    - For every **extra hour** you spend scrolling, your productivity score drops by about **{abs(m):.2f} points** (on a 10-point scale).

    ### 2. A More Human Perspective (The Stress-Relief Hypothesis)
    While the numbers show a drop, we have to ask *why*. Is social media *causing* the low productivity? Or is it something else?
    - **The Coping Mechanism:** It’s possible that when we feel stressed or unproductive, we turn to social media to 'escape' or 'relax.' 
    - This means that for some students, high social media use might be a **symptom** of stress, not the original cause.

    ### 3. Final Takeaway
    In conclusion, while social media usage is statistically linked to decreased productivity in this study, the underlying reasons are likely a complex mix of distraction and stress response. Keeping an eye on your screen time is a great first step toward getting more done!
    """)
    
    st.image("https://img.icons8.com/bubbles/200/000000/student-male.png")

st.markdown("---")
st.caption("✨ Kujenga Final Project | Data Science for a Better Future")
