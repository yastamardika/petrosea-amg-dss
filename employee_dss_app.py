import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor
from io import StringIO
import logging

# Page configuration
st.set_page_config(
    page_title="Employee Performance DSS - SAW Method",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .weight-info {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Database configuration
DB_CONFIG = {
    'host': '10.0.6.3',
    'database': 'petro',
    'user': 'user-name',
    'password': 'strong-password',
    'port': '5432'
}

# Initialize session state
if 'sample_data' not in st.session_state:
    st.session_state.sample_data = None
if 'db_connected' not in st.session_state:
    st.session_state.db_connected = False

class DatabaseConnection:
    def __init__(self, config):
        self.config = config
        self.conn = None
        self.cursor = None
    
    def connect(self):
        """Connect to PostgreSQL database"""
        try:
            self.conn = psycopg2.connect(**self.config)
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            return True
        except Exception as e:
            st.error(f"Database connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from database"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
    
    def execute_query(self, query, params=None):
        """Execute query and return results"""
        try:
            self.cursor.execute(query, params)
            return self.cursor.fetchall()
        except Exception as e:
            st.error(f"Query execution error: {e}")
            return None

    @staticmethod
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def load_employee_performance_data():
        """Load employee performance data from PostgreSQL"""
        db = DatabaseConnection(DB_CONFIG)
        
        if not db.connect():
            return None
        
        try:
            # Fixed query to match actual database schema
            query = """
            WITH employee_metrics AS (
                SELECT 
                    e.employee_id,
                    e.employee_name,
                    d.department_name,
                    p.position_name,
                    
                    -- SHE Metrics (30%)
                    COALESCE(AVG(perf.bipo_compliance_percentage), 80) as she_score,
                    COALESCE(AVG(perf.safety_score), 85) as safety_behavior,
                    COALESCE(AVG(perf.bipo_compliance_percentage * 0.9), 78) as hac_compliance,
                    
                    -- Presence Metrics (20%)
                    COALESCE(AVG(
                        CASE WHEN att.is_absent THEN 0 
                            ELSE 100 END
                    ), 90) as daily_presence,
                    COALESCE(AVG(
                        CASE WHEN att.hours_worked >= 8 THEN 100
                            WHEN att.hours_worked >= 6 THEN 80
                            ELSE 60 END
                    ), 85) as time_card,
                    
                    -- Daily Activity Metrics (25%)
                    COALESCE(AVG(perf.performance_rating), 80) as attitude,
                    COALESCE(AVG(perf.quality_score * 0.95), 82) as appearance,
                    COALESCE(AVG(
                        CASE WHEN p.is_supervisory THEN perf.performance_rating * 1.1
                            ELSE perf.performance_rating END
                    ), 78) as team_leadership,
                    COALESCE(AVG(perf.productivity_score * 0.8), 75) as wag_activity,
                    
                    -- Knowledge Metrics (25%)
                    COALESCE(AVG(perf.productivity_score), 82) as performance_awareness,
                    COALESCE(AVG(
                        CASE WHEN att.is_late THEN perf.performance_rating * 0.9
                            ELSE perf.performance_rating * 1.05 END
                    ), 80) as time_management,
                    COALESCE(AVG(perf.performance_rating * 0.95), 83) as communication,
                    COALESCE(AVG(perf.quality_score), 85) as training_achievement
                    
                FROM dim_employee e
                LEFT JOIN dim_department d ON e.department_id = d.department_id
                LEFT JOIN dim_position p ON e.position_id = p.position_id
                LEFT JOIN fact_performance perf ON e.employee_key = perf.employee_key
                LEFT JOIN fact_attendance att ON e.employee_key = att.employee_key
                WHERE e.is_current = TRUE 
                    AND e.employee_status = 'Active'
                    AND (perf.date_key IS NULL OR perf.date_key IN (
                        SELECT time_id FROM dim_time 
                        WHERE date_actual >= CURRENT_DATE - INTERVAL '90 days'
                    ))
                    AND (att.date_key IS NULL OR att.date_key IN (
                        SELECT time_id FROM dim_time 
                        WHERE date_actual >= CURRENT_DATE - INTERVAL '90 days'
                    ))
                GROUP BY e.employee_id, e.employee_name, d.department_name, p.position_name
            )
            SELECT * FROM employee_metrics
            ORDER BY employee_name
            """
            
            results = db.execute_query(query)
            
            if results:
                df = pd.DataFrame(results)
                # Rename columns to match the expected format in the rest of the code
                df = df.rename(columns={
                    'department_name': 'department',
                    'position_name': 'position'
                })
                return df
            else:
                return None
                
        except Exception as e:
            st.error(f"Error loading employee data: {e}")
            return None
        finally:
            db.disconnect()

def create_sample_data():
    """Create sample employee performance data"""
    np.random.seed(42)
    
    employees = [
        "John Smith", "Maria Garcia", "Ahmed Hassan", "Li Wei", "Sarah Johnson",
        "Carlos Rodriguez", "Fatima Al-Zahra", "Hiroshi Tanaka", "Anna Kowalski", "David Brown"
    ]
    
    departments = ["Heavy Equipment", "Maintenance", "Operations", "Safety", "Engineering"]
    positions = ["Operator", "Supervisor", "Technician", "Engineer", "Specialist"]
    
    data = []
    for i, emp in enumerate(employees):
        # SHE Criteria (30%)
        she_score = np.random.uniform(75, 95)
        safety_behavior = np.random.uniform(80, 100)
        hac_compliance = np.random.uniform(70, 95)
        
        # Presence Criteria (20%)
        daily_presence = np.random.uniform(85, 100)
        time_card = np.random.uniform(80, 98)
        
        # Daily Activity Criteria (25%)
        attitude = np.random.uniform(75, 95)
        appearance = np.random.uniform(80, 100)
        team_leadership = np.random.uniform(70, 90)
        wag_activity = np.random.uniform(60, 85)
        
        # Knowledge Criteria (25%)
        performance_awareness = np.random.uniform(65, 90)
        time_management = np.random.uniform(70, 95)
        communication = np.random.uniform(75, 95)
        training_achievement = np.random.uniform(80, 100)
        
        data.append({
            'employee_id': f'EMP{i+1:03d}',
            'employee_name': emp,
            'department': np.random.choice(departments),
            'position': np.random.choice(positions),
            'she_score': she_score,
            'safety_behavior': safety_behavior,
            'hac_compliance': hac_compliance,
            'daily_presence': daily_presence,
            'time_card': time_card,
            'attitude': attitude,
            'appearance': appearance,
            'team_leadership': team_leadership,
            'wag_activity': wag_activity,
            'performance_awareness': performance_awareness,
            'time_management': time_management,
            'communication': communication,
            'training_achievement': training_achievement
        })
    
    return pd.DataFrame(data)

def normalize_data(df, criteria_columns):
    """Normalize data using min-max normalization (benefit criteria)"""
    normalized_df = df.copy()
    
    for col in criteria_columns:
        max_val = df[col].max()
        min_val = df[col].min()
        if max_val != min_val:
            normalized_df[f'{col}_norm'] = (df[col] - min_val) / (max_val - min_val)
        else:
            normalized_df[f'{col}_norm'] = 1.0
    
    return normalized_df

def calculate_saw_scores(df, weights):
    """Calculate SAW scores for each employee"""
    # Convert all numeric columns to float to avoid Decimal/float multiplication issues
    df_float = df.copy()
    
    # Convert all columns to numeric and then to float
    for col in df_float.columns:
        if col not in ['employee_id', 'employee_name', 'department', 'position']:
            df_float[col] = pd.to_numeric(df_float[col], errors='coerce').astype(float)
    
    # Criteria columns
    criteria_columns = [
        'she_score', 'safety_behavior', 'hac_compliance',
        'daily_presence', 'time_card',
        'attitude', 'appearance', 'team_leadership', 'wag_activity',
        'performance_awareness', 'time_management', 'communication', 'training_achievement'
    ]
    
    # Normalize data
    normalized_df = normalize_data(df_float, criteria_columns)
    
    # Convert weights to float
    weights_float = {k: float(v) for k, v in weights.items()}
    
    # Calculate weighted scores for each category
    normalized_df['she_weighted'] = (
        normalized_df['she_score_norm'] * weights_float['she_score'] +
        normalized_df['safety_behavior_norm'] * weights_float['safety_behavior'] +
        normalized_df['hac_compliance_norm'] * weights_float['hac_compliance']
    ) * 0.30  # SHE weight 30%
    
    normalized_df['presence_weighted'] = (
        normalized_df['daily_presence_norm'] * weights_float['daily_presence'] +
        normalized_df['time_card_norm'] * weights_float['time_card']
    ) * 0.20  # Presence weight 20%
    
    normalized_df['daily_activity_weighted'] = (
        normalized_df['attitude_norm'] * weights_float['attitude'] +
        normalized_df['appearance_norm'] * weights_float['appearance'] +
        normalized_df['team_leadership_norm'] * weights_float['team_leadership'] +
        normalized_df['wag_activity_norm'] * weights_float['wag_activity']
    ) * 0.25  # Daily Activity weight 25%
    
    normalized_df['knowledge_weighted'] = (
        normalized_df['performance_awareness_norm'] * weights_float['performance_awareness'] +
        normalized_df['time_management_norm'] * weights_float['time_management'] +
        normalized_df['communication_norm'] * weights_float['communication'] +
        normalized_df['training_achievement_norm'] * weights_float['training_achievement']
    ) * 0.25  # Knowledge weight 25%
    
    # Calculate final SAW score
    normalized_df['saw_score'] = (
        normalized_df['she_weighted'] +
        normalized_df['presence_weighted'] +
        normalized_df['daily_activity_weighted'] +
        normalized_df['knowledge_weighted']
    )
    
    # Convert to percentage
    normalized_df['saw_score_percentage'] = normalized_df['saw_score'] * 100
    
    return normalized_df

def main():
    st.markdown('<h1 class="main-header">🏢 Employee Performance Evaluation System</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #666;">SAW (Simple Additive Weighting) Method</h2>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Data source selection
    data_source = st.sidebar.selectbox(
        "Select Data Source:",
        ["Sample Data", "Upload CSV File", "Database Connection"]
    )
    
    # Load data
    df = None
    if data_source == "Database Connection":
        if not st.session_state.db_connected:
            with st.spinner("Connecting to database..."):
                db = DatabaseConnection(DB_CONFIG)
                if db.connect():
                    st.session_state.db_connected = True
                    st.success("Database connected successfully!")
                    db.disconnect()
                else:
                    st.error("Failed to connect to database. Using sample data instead.")
                    data_source = "Sample Data"
        
        if st.session_state.db_connected:
            with st.spinner("Loading employee data from database..."):
                df = DatabaseConnection.load_employee_performance_data()
                if df is None:
                    st.error("Failed to load data from database. Using sample data instead.")
                    data_source = "Sample Data"
                    df = create_sample_data()
                else:
                    st.success(f"Loaded {len(df)} employee records from database")
    
    if data_source == "Sample Data":
        if st.session_state.sample_data is None:
            st.session_state.sample_data = create_sample_data()
        df = st.session_state.sample_data.copy()
    elif data_source == "Upload CSV File":
        uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            st.warning("Please upload a CSV file to proceed.")
            return
    
    # If we still don't have data, use sample data
    if df is None:
        df = create_sample_data()
    
    # Weight configuration
    st.sidebar.subheader("📊 Criteria Weights Configuration")
    
    # SHE Weights (30% total)
    st.sidebar.markdown("**SHE (30%)**")
    she_score_weight = st.sidebar.slider("Score Card", 0.0, 1.0, 0.4, 0.05)
    safety_behavior_weight = st.sidebar.slider("Safety Behavior", 0.0, 1.0, 0.3, 0.05)
    hac_compliance_weight = st.sidebar.slider("HAC Compliance", 0.0, 1.0, 0.3, 0.05)
    
    # Presence Weights (20% total)
    st.sidebar.markdown("**Presence (20%)**")
    daily_presence_weight = st.sidebar.slider("Daily Time Sheet", 0.0, 1.0, 0.6, 0.05)
    time_card_weight = st.sidebar.slider("Time Card", 0.0, 1.0, 0.4, 0.05)
    
    # Daily Activity Weights (25% total)
    st.sidebar.markdown("**Daily Activity (25%)**")
    attitude_weight = st.sidebar.slider("Attitude", 0.0, 1.0, 0.3, 0.05)
    appearance_weight = st.sidebar.slider("Appearance", 0.0, 1.0, 0.2, 0.05)
    team_leadership_weight = st.sidebar.slider("Team Member/Leader", 0.0, 1.0, 0.3, 0.05)
    wag_activity_weight = st.sidebar.slider("WAG Activity", 0.0, 1.0, 0.2, 0.05)
    
    # Knowledge Weights (25% total)
    st.sidebar.markdown("**Knowledge (25%)**")
    performance_awareness_weight = st.sidebar.slider("Performance Awareness", 0.0, 1.0, 0.3, 0.05)
    time_management_weight = st.sidebar.slider("Time Management", 0.0, 1.0, 0.25, 0.05)
    communication_weight = st.sidebar.slider("Communication", 0.0, 1.0, 0.25, 0.05)
    training_achievement_weight = st.sidebar.slider("Training Achievement", 0.0, 1.0, 0.2, 0.05)
    
    # Compile weights
    weights = {
        'she_score': she_score_weight,
        'safety_behavior': safety_behavior_weight,
        'hac_compliance': hac_compliance_weight,
        'daily_presence': daily_presence_weight,
        'time_card': time_card_weight,
        'attitude': attitude_weight,
        'appearance': appearance_weight,
        'team_leadership': team_leadership_weight,
        'wag_activity': wag_activity_weight,
        'performance_awareness': performance_awareness_weight,
        'time_management': time_management_weight,
        'communication': communication_weight,
        'training_achievement': training_achievement_weight
    }
    
    # Calculate SAW scores
    result_df = calculate_saw_scores(df, weights)
    
    # Sort by SAW score
    result_df = result_df.sort_values('saw_score_percentage', ascending=False).reset_index(drop=True)
    result_df['rank'] = range(1, len(result_df) + 1)
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Employees", len(result_df))
    with col2:
        st.metric("Average Score", f"{result_df['saw_score_percentage'].mean():.1f}%")
    with col3:
        st.metric("Highest Score", f"{result_df['saw_score_percentage'].max():.1f}%")
    with col4:
        st.metric("Lowest Score", f"{result_df['saw_score_percentage'].min():.1f}%")
    
    # Display weight information
    st.markdown('<div class="weight-info">', unsafe_allow_html=True)
    st.markdown("### 📋 Current Weight Configuration")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**SHE (30%)**")
        st.write(f"• Score Card: {she_score_weight:.2f}")
        st.write(f"• Safety Behavior: {safety_behavior_weight:.2f}")
        st.write(f"• HAC Compliance: {hac_compliance_weight:.2f}")
    
    with col2:
        st.markdown("**Presence (20%)**")
        st.write(f"• Daily Time Sheet: {daily_presence_weight:.2f}")
        st.write(f"• Time Card: {time_card_weight:.2f}")
    
    with col3:
        st.markdown("**Daily Activity (25%)**")
        st.write(f"• Attitude: {attitude_weight:.2f}")
        st.write(f"• Appearance: {appearance_weight:.2f}")
        st.write(f"• Team Member/Leader: {team_leadership_weight:.2f}")
        st.write(f"• WAG Activity: {wag_activity_weight:.2f}")
    
    with col4:
        st.markdown("**Knowledge (25%)**")
        st.write(f"• Performance Awareness: {performance_awareness_weight:.2f}")
        st.write(f"• Time Management: {time_management_weight:.2f}")
        st.write(f"• Communication: {communication_weight:.2f}")
        st.write(f"• Training Achievement: {training_achievement_weight:.2f}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["🏆 Rankings", "📊 Analysis", "📈 Visualizations", "📋 Detailed Results"])
    
    with tab1:
        st.subheader("Employee Performance Rankings")
        
        # Top performers
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🥇 Top 5 Performers")
            top_5 = result_df.head(5)[['rank', 'employee_name', 'department', 'saw_score_percentage']]
            for _, row in top_5.iterrows():
                st.markdown(f"**#{row['rank']} {row['employee_name']}**")
                st.markdown(f"*{row['department']} - Score: {row['saw_score_percentage']:.1f}%*")
                st.markdown("---")
        
        with col2:
            st.markdown("#### 📈 Performance Distribution")
            fig = px.histogram(result_df, x='saw_score_percentage', nbins=10, 
                             title="Score Distribution",
                             labels={'saw_score_percentage': 'SAW Score (%)', 'count': 'Number of Employees'})
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Performance Analysis")
        
        # Department analysis
        dept_analysis = result_df.groupby('department').agg({
            'saw_score_percentage': ['mean', 'std', 'count'],
            'she_weighted': 'mean',
            'presence_weighted': 'mean',
            'daily_activity_weighted': 'mean',
            'knowledge_weighted': 'mean'
        }).round(2)
        dept_analysis.columns = ['Avg Score', 'Std Dev', 'Count', 'SHE Avg', 'Presence Avg', 'Activity Avg', 'Knowledge Avg']
        
        st.markdown("#### 🏢 Department Performance Analysis")
        st.dataframe(dept_analysis, use_container_width=True)
        
        # Correlation analysis
        st.markdown("#### 🔗 Criteria Correlation")
        criteria_cols = ['she_weighted', 'presence_weighted', 'daily_activity_weighted', 'knowledge_weighted']
        corr_matrix = result_df[criteria_cols].corr()
        
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                       title="Correlation Matrix of Performance Categories")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Performance Visualizations")
        
        # Scatter plot
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(result_df, x='she_weighted', y='knowledge_weighted', 
                           size='saw_score_percentage', color='department',
                           hover_name='employee_name',
                           title="SHE vs Knowledge Performance")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(result_df, x='department', y='saw_score_percentage',
                        title="Score Distribution by Department")
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Radar chart for top performer
        st.markdown("#### 🎯 Top Performer Profile")
        top_performer = result_df.iloc[0]
        
        categories = ['SHE', 'Presence', 'Daily Activity', 'Knowledge']
        values = [
            top_performer['she_weighted'] * (100/0.30),
            top_performer['presence_weighted'] * (100/0.20),
            top_performer['daily_activity_weighted'] * (100/0.25),
            top_performer['knowledge_weighted'] * (100/0.25)
        ]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=top_performer['employee_name']
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title=f"Performance Profile: {top_performer['employee_name']}"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Detailed Results")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            dept_filter = st.selectbox("Filter by Department", 
                                     ["All"] + list(result_df['department'].unique()))
        
        with col2:
            min_score = st.slider("Minimum Score (%)", 0, 100, 0)
        
        with col3:
            max_rank = st.slider("Top N Employees", 1, len(result_df), len(result_df))
        
        # Apply filters
        filtered_df = result_df.copy()
        if dept_filter != "All":
            filtered_df = filtered_df[filtered_df['department'] == dept_filter]
        filtered_df = filtered_df[filtered_df['saw_score_percentage'] >= min_score]
        filtered_df = filtered_df.head(max_rank)
        
        # Display detailed results
        display_columns = [
            'rank', 'employee_name', 'department', 'position',
            'saw_score_percentage', 'she_weighted', 'presence_weighted',
            'daily_activity_weighted', 'knowledge_weighted'
        ]
        
        formatted_df = filtered_df[display_columns].copy()
        formatted_df.columns = [
            'Rank', 'Employee Name', 'Department', 'Position',
            'SAW Score (%)', 'SHE Score', 'Presence Score',
            'Daily Activity Score', 'Knowledge Score'
        ]
        
        # Format numeric columns
        numeric_cols = ['SAW Score (%)', 'SHE Score', 'Presence Score', 'Daily Activity Score', 'Knowledge Score']
        for col in numeric_cols:
            if col == 'SAW Score (%)':
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.1f}%")
            else:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.3f}")
        
        st.dataframe(formatted_df, use_container_width=True)
        
        # Download button
        csv = result_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Results as CSV",
            data=csv,
            file_name=f"employee_performance_saw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()