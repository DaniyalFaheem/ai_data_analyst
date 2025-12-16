"""
AI Data Analyst - Complete Application with Ultra-Powered DataGPT
No Debug Icons - Production Ready
"""

import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import base64
import io
import uuid
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier, IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

SERVER_DATA_CACHE = {}

def get_dataframe(session_id):
    """
    Safely retrieve DataFrame from cache, preferring cleaned over original.
    Uses explicit None checks to avoid DataFrame truth value ambiguity.
    """
    if session_id not in SERVER_DATA_CACHE:
        return None
    cache = SERVER_DATA_CACHE[session_id]
    cleaned = cache.get('cleaned')
    if cleaned is not None:
        return cleaned
    return cache.get('original')

CUSTOM_STYLE = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    * { font-family: 'Inter', sans-serif ! important; }
    
    body { 
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
        background-attachment: fixed; 
        margin:  0; 
        padding:  0; 
    }
    
    .main-container { 
        background: rgba(255, 255, 255, 0.95); 
        border-radius:  20px; 
        padding: 30px; 
        margin-top: 20px; 
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3); 
    }
    
    .glass-card { 
        background: white ! important; 
        border-radius: 15px ! important; 
        border: 1px solid #e0e0e0 ! important; 
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1) !important; 
        transition: transform 0.3s ease ! important; 
    }
    
    .glass-card:hover { 
        transform: translateY(-5px) !important; 
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15) !important; 
    }
    
    .kpi-card { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
        border-radius: 15px; 
        padding: 25px; 
        text-align: center; 
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.4); 
        transition: transform 0.3s ease; 
    }
    
    .kpi-card:hover { transform: scale(1.05); }
    
    .kpi-icon { font-size: 3rem; margin-bottom: 10px; color: white; }
    .kpi-value { font-size: 2.5rem; font-weight: 700; color: white; margin:  10px 0; }
    . kpi-label { font-size: 0.9rem; color: rgba(255, 255, 255, 0.9); text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
    
    .upload-zone { 
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.15) 0%, rgba(255, 255, 255, 0.1) 100%); 
        border:  3px dashed rgba(255, 255, 255, 0.6); 
        border-radius:  20px; 
        padding: 50px 40px; 
        text-align: center; 
        cursor: pointer; 
        transition: all 0.3s ease; 
        backdrop-filter: blur(10px); 
    }
    
    .upload-zone:hover { 
        border-color: #ffd93d; 
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.25) 0%, rgba(255, 255, 255, 0.2) 100%); 
        transform: scale(1.02); 
        box-shadow: 0 10px 40px rgba(255, 217, 61, 0.3); 
    }
    
    .custom-nav { 
        background: linear-gradient(135deg, #0a1929 0%, #1a2942 100%) !important; 
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5) !important; 
        padding: 15px 30px !important; 
        border-bottom: 2px solid rgba(255, 255, 255, 0.1) !important; 
    }
    
    .nav-link { 
        color: #ffffff !important; 
        font-weight: 600 !important; 
        margin:  0 15px !important; 
        font-size: 1.05rem !important; 
        transition: all 0.3s ease !important; 
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3) !important; 
        letter-spacing: 0.3px !important; 
    }
    
    .nav-link:hover { 
        color:  #ffd93d !important; 
        transform: translateY(-2px) !important; 
        text-shadow: 0 3px 6px rgba(255, 217, 61, 0.4) !important; 
    }
    
    .brand-logo { 
        font-size: 1.7rem ! important; 
        font-weight: 800 !important; 
        color: #ffffff !important; 
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.4) !important; 
        letter-spacing: 0.5px !important; 
    }
    
    .btn-custom { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; 
        border:  none !important; 
        border-radius: 10px !important; 
        padding: 12px 30px ! important; 
        font-weight: 600 !important; 
        color: white !important; 
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important; 
        transition: all 0.3s ease !important; 
    }
    
    .btn-custom:hover { 
        transform: translateY(-3px) !important; 
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5) !important; 
        color: white !important; 
    }
    
    .chat-container { 
        background: #f8f9fa; 
        border-radius: 15px; 
        padding: 20px; 
        max-height: 500px; 
        overflow-y: auto; 
        border:  1px solid #e0e0e0; 
    }
    
    .chat-message-user { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
        padding: 12px 18px; 
        border-radius:  18px 18px 5px 18px; 
        margin: 10px 0; 
        max-width: 80%; 
        margin-left: auto; 
        color:  white; 
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3); 
    }
    
    .chat-message-bot { 
        background:  white; 
        padding: 12px 18px; 
        border-radius: 18px 18px 18px 5px; 
        margin: 10px 0; 
        max-width: 80%; 
        color: #333; 
        border: 1px solid #e0e0e0; 
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05); 
    }
    
    .section-title { 
        font-size: 1.8rem; 
        font-weight: 700; 
        color: #2c3e50; 
        margin-bottom: 25px; 
        position: relative; 
        padding-bottom: 15px; 
    }
    
    .section-title:: after { 
        content: ''; 
        position: absolute; 
        bottom: 0; 
        left: 0; 
        width: 80px; 
        height: 4px; 
        background: linear-gradient(90deg, #667eea, #764ba2); 
        border-radius: 2px; 
    }
    
    ::-webkit-scrollbar { width: 10px; }
    : :-webkit-scrollbar-track { background: #f1f1f1; border-radius: 10px; }
    ::-webkit-scrollbar-thumb { background:  linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; }
    : :-webkit-scrollbar-thumb: hover { background: linear-gradient(135deg, #764ba2 0%, #667eea 100%); }
    
    .alert-custom { border-radius: 12px; border: none; padding: 15px 20px; font-weight:  500; }
    .table-container { background: white; border-radius: 15px; padding: 20px; overflow-x: auto; border: 1px solid #e0e0e0; }
    
    label { color: #2c3e50 !important; font-weight: 600 !important; }
    
    /* Fix for AutoML scrolling issue - prevent scroll anchor behavior */
    #automl-results-container {
        overflow-anchor: none !important;
    }
    
    /* Prevent graphs from causing scroll jumps */
    .js-plotly-plot, .plotly {
        overflow: hidden !important;
    }
    
    /* ========== REMOVE ALL DEBUG/DEV ICONS ========== */
    ._dash-undo-redo, 
    ._dash-loading, 
    . dash-debug-menu, 
    ._dash-error-menu, 
    . dash-callback-error,
    [class*="copilot"], 
    [id*="copilot"], 
    [data-copilot], 
    . github-copilot-icon,
    div[style*="position: fixed"][style*="bottom"][style*="right"],
    body > div: last-child[style*="position: fixed"] { 
        display: none !important; 
        visibility: hidden !important; 
        opacity: 0 !important;
        pointer-events: none !important;
    }
    
    /* Hide any floating dev tools */
    #_dash-app-content > div:last-child[style*="position: fixed"] {
        display: none !important;
    }
"""

app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.BOOTSTRAP], 
    suppress_callback_exceptions=True, 
    title="AI Data Analyst"
)

app.index_string = '''
<! DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>''' + CUSTOM_STYLE + '''</style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def parse_upload_contents(contents, filename):
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        if 'csv' in filename.lower():
            return pd.read_csv(io. StringIO(decoded.decode('utf-8')))
        return None
    except Exception as e: 
        print(f"Error:  {e}")
        return None

def auto_clean_data(df):
    df_clean = df.copy()
    df_clean = df_clean.drop_duplicates()
    
    for col in df_clean.columns:
        if df_clean[col]. dtype == 'object':
            try:
                df_clean[col] = pd.to_datetime(df_clean[col])
            except:
                pass
    
    for col in df_clean.columns:
        if df_clean[col].dtype in ['float64', 'int64']: 
            df_clean[col]. fillna(df_clean[col].median(), inplace=True)
        elif df_clean[col].dtype == 'object':
            mode_series = df_clean[col].mode()
            if not mode_series.empty:
                df_clean[col]. fillna(mode_series[0], inplace=True)
    
    return df_clean

def get_health_stats(df):
    total_cells = df.shape[0] * df.shape[1]
    missing = int(df.isnull().sum().sum())
    duplicates = int(df.duplicated().sum())
    return {'missing':  missing, 'duplicates': duplicates, 'total_cells':  total_cells}

def create_smart_pie_chart(df, column):
    try:
        if df[column]. dtype == 'datetime64[ns]':
            data = df[column].dt.year.value_counts().head(10)
            labels = [str(x) for x in data.index]
            title = f"{column} Distribution by Year"
        elif df[column].dtype in ['float64', 'int64']: 
            try:
                binned = pd.cut(df[column]. dropna(), bins=10)
                data = binned.value_counts().head(10)
                labels = [f"{interval. left:. 2f} - {interval.right:.2f}" for interval in data.index]
            except:
                data = df[column].value_counts().head(10)
                labels = [str(x) for x in data.index]
            title = f"{column} Distribution"
        else:
            top_10 = df[column].value_counts().head(10)
            others_count = df[column].value_counts().iloc[10:].sum()
            if others_count > 0:
                top_10 = pd.concat([top_10, pd.Series({'Others': others_count})])
            data = top_10
            labels = [str(x) for x in data.index]
            title = f"{column} Distribution"
        
        fig = px.pie(values=data.values, names=labels, title=title, color_discrete_sequence=px.colors.qualitative.Set2, hole=0.4)
        fig.update_layout(template='plotly_white', paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter', color='#2c3e50'))
        return fig
    except: 
        return go.Figure()

def create_correlation_heatmap(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Not enough numerical columns", showarrow=False, font=dict(size=20, color='#2c3e50'))
        fig.update_layout(template='plotly_white')
        return fig
    
    corr = df[numeric_cols].corr()
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='RdBu', zmid=0, text=corr.values, texttemplate='%{text:.2f}'))
    fig.update_layout(title="Correlation Heatmap", template='plotly_white', height=600, font=dict(family='Inter', color='#2c3e50'))
    return fig

def create_distribution_plot(df, column):
    if df[column].dtype not in ['float64', 'int64']:
        fig = go.Figure()
        fig.add_annotation(text=f"{column} is not numerical", showarrow=False, font=dict(size=20, color='#2c3e50'))
        fig.update_layout(template='plotly_white')
        return fig
    
    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], subplot_titles=(f'{column} Distribution', f'{column} Boxplot'), vertical_spacing=0.1)
    fig.add_trace(go. Histogram(x=df[column], name='Distribution', marker_color='#667eea', opacity=0.8), row=1, col=1)
    fig.add_trace(go.Box(x=df[column], name='Boxplot', marker_color='#764ba2'), row=2, col=1)
    fig.update_layout(template='plotly_white', height=600, showlegend=False, font=dict(family='Inter', color='#2c3e50'))
    return fig

def detect_task_type(df, target_column):
    unique_values = df[target_column].nunique()
    return 'classification' if (df[target_column].dtype == 'object' or unique_values < 20) else 'regression'

def train_ml_model(df, target_column):
    """Enhanced ML training with multiple models and automatic best model selection.
    
    Optimized for fast training with reduced n_estimators while maintaining good accuracy.
    """
    df_model = df.copy()
    label_encoders = {}
    
    # Encode categorical columns
    for col in df_model.columns:
        if df_model[col].dtype == 'object':
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col].astype(str))
            label_encoders[col] = le
    
    # Handle missing values
    df_model = df_model.fillna(df_model.median(numeric_only=True))
    for col in df_model.columns:
        if df_model[col].isnull().any():
            df_model[col] = df_model[col].fillna(df_model[col].mode()[0] if not df_model[col].mode().empty else 0)
    
    X = df_model.drop(columns=[target_column])
    y = df_model[target_column]
    
    # Scale features for better performance
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y if detect_task_type(df, target_column) == 'classification' and y.nunique() > 1 else None)
    
    task_type = detect_task_type(df, target_column)
    
    best_model = None
    best_score = -float('inf')
    best_model_name = ""
    all_results = []
    
    if task_type == 'classification':
        # Try multiple classifiers and pick the best one
        # Optimized for speed with reduced n_estimators
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=5, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=min(5, len(X_train)//2) if len(X_train) > 2 else 1),
        }
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                score = accuracy_score(y_test, model.predict(X_test))
                all_results.append({'model': name, 'score': score})
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_model_name = name
            except Exception:
                continue
        
        metric_name = "Accuracy"
    else:
        # Try multiple regressors and pick the best one
        # Optimized for speed with reduced n_estimators
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=5, random_state=42),
            'Ridge': Ridge(alpha=1.0),
        }
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                score = r2_score(y_test, model.predict(X_test))
                all_results.append({'model': name, 'score': score})
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_model_name = name
            except Exception:
                continue
        
        metric_name = "R² Score"
    
    # Get feature importance
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
    else:
        # For models without feature_importances_, use coefficient-based importance
        if hasattr(best_model, 'coef_'):
            importances = np.abs(best_model.coef_).flatten() if len(best_model.coef_.shape) > 1 else np.abs(best_model.coef_)
            if len(importances) == len(X.columns):
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': importances
                }).sort_values('importance', ascending=False)
            else:
                feature_importance = pd.DataFrame({'feature': X.columns, 'importance': [1/len(X.columns)] * len(X.columns)})
        else:
            feature_importance = pd.DataFrame({'feature': X.columns, 'importance': [1/len(X.columns)] * len(X.columns)})
    
    return {
        'task_type': task_type,
        'score': best_score,
        'metric_name': metric_name,
        'feature_importance': feature_importance,
        'best_model_name': best_model_name,
        'all_results': all_results
    }

def create_feature_importance_chart(feature_importance):
    fig = px.bar(feature_importance. head(15), x='importance', y='feature', orientation='h', title='Top 15 Feature Importance', color='importance', color_continuous_scale=[[0, '#667eea'], [1, '#764ba2']])
    fig.update_layout(template='plotly_white', height=500, font=dict(family='Inter', color='#2c3e50'))
    return fig

def perform_anomaly_detection(df, column):
    if df[column].dtype not in ['float64', 'int64']:
        return None, "Column must be numerical"
    
    data = df[[column]].dropna()
    if data.empty or len(data) < 10:
        return None, "Not enough data"
    
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    predictions = iso_forest.fit_predict(data)
    
    anomalies = data[predictions == -1]
    normal = data[predictions == 1]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=normal. index, y=normal[column], mode='markers', name='Normal', marker=dict(color='#667eea', size=6, opacity=0.7)))
    fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies[column], mode='markers', name='Anomaly', marker=dict(color='#ff6b9d', size=10, symbol='x')))
    fig.update_layout(title=f'Anomaly Detection for {column}', template='plotly_white', height=400, font=dict(family='Inter', color='#2c3e50'))
    
    return fig, f"Found {len(anomalies)} anomalies out of {len(data)} data points"

def perform_forecasting(df, column, periods=10):
    if df[column]. dtype not in ['float64', 'int64']:
        return None, "Column must be numerical"
    
    data = df[[column]].dropna().reset_index(drop=True)
    if data.empty or len(data) < 10:
        return None, "Not enough data"
    
    data['index'] = range(len(data))
    X = data[['index']]
    y = data[column]
    
    model = GradientBoostingRegressor(n_estimators=30, random_state=42)
    model.fit(X, y)
    
    future_indices = np.arange(len(data), len(data) + periods).reshape(-1, 1)
    predictions = model.predict(future_indices)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['index'], y=data[column], mode='lines', name='Historical', line=dict(color='#667eea', width=3)))
    fig.add_trace(go.Scatter(x=list(range(len(data), len(data) + periods)), y=predictions, mode='lines+markers', name='Forecast', line=dict(color='#764ba2', dash='dash', width=3)))
    fig.update_layout(title=f'Forecast for {column}', template='plotly_white', height=400, font=dict(family='Inter', color='#2c3e50'))
    
    mse = mean_squared_error(y[-10:], model.predict(X[-10:]))
    return fig, f"Forecast RMSE: {np.sqrt(mse):.2f}"

def monte_carlo_simulation(df, column, n_simulations=200):
    if df[column].dtype not in ['float64', 'int64']:
        return None, "Column must be numerical"
    
    data = df[column].dropna()
    if data.empty or len(data) < 10:
        return None, "Not enough data"
    
    mean = data.mean()
    std = data.std()
    simulations = np.random.normal(mean, std, (n_simulations, len(data)))
    
    fig = go.Figure()
    for i in range(min(50, n_simulations)):
        fig.add_trace(go. Scatter(y=simulations[i], mode='lines', line=dict(width=0.5, color='rgba(102, 126, 234, 0.3)'), showlegend=False, hoverinfo='skip'))
    
    fig.add_trace(go.Scatter(y=data. values, mode='lines', name='Actual', line=dict(color='#764ba2', width=3)))
    fig.update_layout(title=f'Monte Carlo Simulation for {column}', template='plotly_white', height=400, font=dict(family='Inter', color='#2c3e50'))
    return fig, f"Ran {n_simulations} simulations"

def create_sankey_diagram(df):
    numeric_cols = max(len(df.select_dtypes(include=[np.number]).columns), 1)
    categorical_cols = max(len(df.select_dtypes(include=['object']).columns), 1)
    datetime_cols = max(len(df.select_dtypes(include=['datetime64']).columns), 1)
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(color="#2c3e50", width=0.5), label=["Raw Data", "Numeric", "Categorical", "DateTime", "Processed Data"], color=["#667eea", "#764ba2", "#ffd93d", "#ff6b9d", "#00d9a5"]),
        link=dict(source=[0, 0, 0, 1, 2, 3], target=[1, 2, 3, 4, 4, 4], value=[numeric_cols, categorical_cols, datetime_cols, numeric_cols, categorical_cols, datetime_cols])
    )])
    fig.update_layout(title="Data Processing Lineage", template='plotly_white', height=400, font=dict(family='Inter', color='#2c3e50'))
    return fig

def generate_professional_report(df, filename="Dataset"):
    """Generate a professional, eye-catching HTML report with all visualizations and data analysis"""
    
    # Get data statistics
    total_rows = len(df)
    total_cols = len(df.columns)
    total_cells = total_rows * total_cols
    missing_values = int(df.isnull().sum().sum())
    duplicates = int(df.duplicated().sum())
    memory_usage = df.memory_usage(deep=True).sum() / 1024**2
    completeness = ((total_cells - missing_values) / total_cells * 100) if total_cells > 0 else 0
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Generate timestamp
    report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    
    # Create visualizations as base64 images
    charts_html = ""
    
    # 1. Create distribution charts for top categorical columns
    for col in categorical_cols[:3]:
        try:
            value_counts = df[col].value_counts().head(10)
            fig = px.pie(values=value_counts.values, names=value_counts.index.astype(str), 
                        title=f"Distribution: {col}", hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(template='plotly_white', height=400, font=dict(family='Inter'))
            chart_html = fig.to_html(full_html=False, include_plotlyjs=False)
            charts_html += f'<div class="chart-container">{chart_html}</div>'
        except:
            pass
    
    # 2. Create histogram for top numeric columns
    for col in numeric_cols[:3]:
        try:
            fig = px.histogram(df, x=col, title=f"Distribution: {col}", 
                              color_discrete_sequence=['#667eea'])
            fig.update_layout(template='plotly_white', height=400, font=dict(family='Inter'))
            chart_html = fig.to_html(full_html=False, include_plotlyjs=False)
            charts_html += f'<div class="chart-container">{chart_html}</div>'
        except:
            pass
    
    # 3. Correlation heatmap if enough numeric columns
    if len(numeric_cols) >= 2:
        try:
            corr = df[numeric_cols].corr()
            fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, 
                                           colorscale='RdBu', zmid=0))
            fig.update_layout(title="Correlation Heatmap", template='plotly_white', 
                            height=500, font=dict(family='Inter'))
            chart_html = fig.to_html(full_html=False, include_plotlyjs=False)
            charts_html += f'<div class="chart-container full-width">{chart_html}</div>'
        except:
            pass
    
    # Generate column details table
    column_details = ""
    for col in df.columns:
        dtype = str(df[col].dtype)
        unique = df[col].nunique()
        missing = df[col].isnull().sum()
        missing_pct = (missing / total_rows * 100) if total_rows > 0 else 0
        
        if df[col].dtype in ['float64', 'int64']:
            stats = f"Min: {df[col].min():.2f}, Max: {df[col].max():.2f}, Mean: {df[col].mean():.2f}"
        else:
            top_val = df[col].mode()[0] if not df[col].mode().empty else 'N/A'
            stats = f"Most Common: {top_val}"
        
        status_class = "status-good" if missing_pct < 5 else "status-warning" if missing_pct < 20 else "status-bad"
        column_details += f'''
        <tr>
            <td><strong>{col}</strong></td>
            <td><span class="badge badge-type">{dtype}</span></td>
            <td>{unique:,}</td>
            <td><span class="{status_class}">{missing:,} ({missing_pct:.1f}%)</span></td>
            <td>{stats}</td>
        </tr>'''
    
    # Professional HTML Report
    html_report = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Analysis Report - {filename}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}
        
        body {{
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #667eea 100%);
            min-height: 100vh;
            padding: 40px 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .header {{
            background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.9) 100%);
            border-radius: 24px;
            padding: 50px;
            margin-bottom: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            text-align: center;
            position: relative;
            overflow: hidden;
        }}
        
        .header::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 6px;
            background: linear-gradient(90deg, #667eea, #764ba2, #ff6b9d, #ffd93d);
        }}
        
        .header-icon {{
            font-size: 4rem;
            margin-bottom: 20px;
        }}
        
        .header h1 {{
            font-size: 2.8rem;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 10px;
        }}
        
        .header .subtitle {{
            font-size: 1.2rem;
            color: #666;
            margin-bottom: 5px;
        }}
        
        .header .date {{
            font-size: 0.95rem;
            color: #888;
        }}
        
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .kpi-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 20px;
            padding: 30px;
            text-align: center;
            box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4);
            transition: transform 0.3s ease;
        }}
        
        .kpi-card:hover {{
            transform: translateY(-5px);
        }}
        
        .kpi-card.success {{
            background: linear-gradient(135deg, #00d9a5 0%, #00b894 100%);
            box-shadow: 0 10px 40px rgba(0, 217, 165, 0.4);
        }}
        
        .kpi-card.warning {{
            background: linear-gradient(135deg, #ffd93d 0%, #f39c12 100%);
            box-shadow: 0 10px 40px rgba(255, 217, 61, 0.4);
        }}
        
        .kpi-card.info {{
            background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
            box-shadow: 0 10px 40px rgba(116, 185, 255, 0.4);
        }}
        
        .kpi-icon {{
            font-size: 2.5rem;
            margin-bottom: 10px;
        }}
        
        .kpi-value {{
            font-size: 2.2rem;
            font-weight: 800;
            color: white;
            margin-bottom: 5px;
        }}
        
        .kpi-label {{
            font-size: 0.85rem;
            color: rgba(255,255,255,0.9);
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 600;
        }}
        
        .section {{
            background: rgba(255,255,255,0.95);
            border-radius: 20px;
            padding: 40px;
            margin-bottom: 30px;
            box-shadow: 0 15px 50px rgba(0,0,0,0.2);
        }}
        
        .section-title {{
            font-size: 1.8rem;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 30px;
            padding-bottom: 15px;
            border-bottom: 3px solid;
            border-image: linear-gradient(90deg, #667eea, #764ba2) 1;
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        
        .section-title span {{
            font-size: 1.5rem;
        }}
        
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px;
        }}
        
        .chart-container {{
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.08);
            border: 1px solid #eee;
        }}
        
        .chart-container.full-width {{
            grid-column: 1 / -1;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        th:first-child {{
            border-radius: 10px 0 0 0;
        }}
        
        th:last-child {{
            border-radius: 0 10px 0 0;
        }}
        
        td {{
            padding: 15px;
            border-bottom: 1px solid #eee;
            color: #333;
        }}
        
        tr:hover {{
            background: #f8f9ff;
        }}
        
        .badge {{
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
        }}
        
        .badge-type {{
            background: linear-gradient(135deg, #e8f4fd 0%, #d4e8f7 100%);
            color: #0984e3;
        }}
        
        .status-good {{
            color: #00b894;
            font-weight: 600;
        }}
        
        .status-warning {{
            color: #f39c12;
            font-weight: 600;
        }}
        
        .status-bad {{
            color: #e74c3c;
            font-weight: 600;
        }}
        
        .quality-meter {{
            background: #eee;
            border-radius: 10px;
            height: 20px;
            overflow: hidden;
            margin: 20px 0;
        }}
        
        .quality-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            border-radius: 10px;
            transition: width 1s ease;
        }}
        
        .footer {{
            text-align: center;
            padding: 30px;
            color: rgba(255,255,255,0.8);
            font-size: 0.9rem;
        }}
        
        .footer strong {{
            color: white;
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        
        .summary-item {{
            background: linear-gradient(135deg, #f8f9ff 0%, #fff 100%);
            border-radius: 12px;
            padding: 20px;
            border-left: 4px solid #667eea;
        }}
        
        .summary-item h4 {{
            color: #667eea;
            margin-bottom: 8px;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .summary-item p {{
            color: #2c3e50;
            font-size: 1.1rem;
            font-weight: 600;
        }}
        
        @media print {{
            body {{
                background: white;
                padding: 20px;
            }}
            .kpi-card, .section {{
                box-shadow: none;
                border: 1px solid #ddd;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div class="header-icon">📊</div>
            <h1>Professional Data Analysis Report</h1>
            <p class="subtitle">Comprehensive Analysis of <strong>{filename}</strong></p>
            <p class="date">Generated on {report_date}</p>
        </div>
        
        <!-- Executive Summary KPIs -->
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-icon">📊</div>
                <div class="kpi-value">{total_rows:,}</div>
                <div class="kpi-label">Total Records</div>
            </div>
            <div class="kpi-card info">
                <div class="kpi-icon">📋</div>
                <div class="kpi-value">{total_cols}</div>
                <div class="kpi-label">Data Features</div>
            </div>
            <div class="kpi-card success">
                <div class="kpi-icon">✅</div>
                <div class="kpi-value">{completeness:.1f}%</div>
                <div class="kpi-label">Data Completeness</div>
            </div>
            <div class="kpi-card warning">
                <div class="kpi-icon">⚠️</div>
                <div class="kpi-value">{missing_values:,}</div>
                <div class="kpi-label">Missing Values</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">🔄</div>
                <div class="kpi-value">{duplicates:,}</div>
                <div class="kpi-label">Duplicate Rows</div>
            </div>
            <div class="kpi-card info">
                <div class="kpi-icon">💾</div>
                <div class="kpi-value">{memory_usage:.2f} MB</div>
                <div class="kpi-label">Memory Usage</div>
            </div>
        </div>
        
        <!-- Data Quality Section -->
        <div class="section">
            <h2 class="section-title"><span>🎯</span> Data Quality Score</h2>
            <div class="quality-meter">
                <div class="quality-fill" style="width: {completeness}%;"></div>
            </div>
            <p style="text-align: center; color: #666; margin-top: 10px;">
                Your data quality score is <strong style="color: #667eea; font-size: 1.2rem;">{completeness:.1f}%</strong>
            </p>
            
            <div class="summary-grid">
                <div class="summary-item">
                    <h4>Numeric Columns</h4>
                    <p>{len(numeric_cols)} features</p>
                </div>
                <div class="summary-item">
                    <h4>Categorical Columns</h4>
                    <p>{len(categorical_cols)} features</p>
                </div>
                <div class="summary-item">
                    <h4>DateTime Columns</h4>
                    <p>{len(datetime_cols)} features</p>
                </div>
                <div class="summary-item">
                    <h4>Total Cells</h4>
                    <p>{total_cells:,} data points</p>
                </div>
            </div>
        </div>
        
        <!-- Visualizations Section -->
        <div class="section">
            <h2 class="section-title"><span>📈</span> Data Visualizations</h2>
            <div class="charts-grid">
                {charts_html}
            </div>
        </div>
        
        <!-- Column Details Section -->
        <div class="section">
            <h2 class="section-title"><span>📋</span> Column Analysis</h2>
            <table>
                <thead>
                    <tr>
                        <th>Column Name</th>
                        <th>Data Type</th>
                        <th>Unique Values</th>
                        <th>Missing</th>
                        <th>Statistics</th>
                    </tr>
                </thead>
                <tbody>
                    {column_details}
                </tbody>
            </table>
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <p>🤖 Generated by <strong>AI Data Analyst</strong> | Professional Data Analysis Platform</p>
            <p style="margin-top: 5px; font-size: 0.8rem;">© {datetime.now().year} All Rights Reserved</p>
        </div>
    </div>
</body>
</html>'''
    
    return html_report

def parse_datagpt_query(query, df, filename=""):
    """ULTRA-SMART DataGPT - Handles ALL question variations with intelligent matching"""
    q = query. lower().strip()
    
    # ========== SMART COLUMN DETECTION ==========
    def find_column(query_text, df_columns):
        """Find column with fuzzy matching, partial matching, case-insensitive"""
        query_text = query_text.lower()
        
        # Exact match (case-insensitive)
        for col in df_columns:
            if col.lower() == query_text:
                return col
        
        # Partial match - check if query contains column name or vice versa
        for col in df_columns:
            col_lower = col.lower()
            # Remove common separators for matching
            col_clean = col_lower.replace('_', ' ').replace('-', ' ')
            query_clean = query_text.replace('_', ' ').replace('-', ' ')
            
            if col_lower in query_text or query_text in col_lower:
                return col
            if col_clean in query_clean or query_clean in col_clean:
                return col
            
            # Word-by-word matching for multi-word columns
            col_words = col_clean.split()
            query_words = query_clean.split()
            if any(word in query_words for word in col_words if len(word) > 2):
                return col
        
        return None
    
    def detect_intent_and_column(query_text, df_columns):
        """Detect what user wants to know and which column they're asking about"""
        intents = {
            'mean': ['mean', 'average', 'avg'],
            'max': ['max', 'maximum', 'highest', 'largest', 'biggest', 'top'],
            'min': ['min', 'minimum', 'lowest', 'smallest', 'bottom'],
            'median': ['median', 'middle'],
            'sum': ['sum', 'total', 'add up'],
            'count': ['count', 'how many', 'number of'],
            'unique': ['unique', 'distinct', 'different'],
            'analyze': ['analyze', 'analysis', 'tell me about', 'describe', 'info about', 'information about'],
            'show': ['show', 'display', 'list', 'give me'],
            'std': ['std', 'standard deviation', 'variance', 'variability'],
            'mode': ['mode', 'most common', 'most frequent', 'popular']
        }
        
        detected_intent = None
        detected_column = None
        
        # Detect intent
        for intent, keywords in intents.items():
            if any(keyword in query_text for keyword in keywords):
                detected_intent = intent
                break
        
        # Try to find column in query - check if column name appears in the query text
        query_lower = query_text.lower()
        query_words = query_lower.replace('_', ' ').replace('-', ' ').replace("'s", "").split()
        
        # Helper to normalize words (handle plurals, etc.)
        def normalize_word(word):
            word = word.lower().strip()
            # Remove common suffixes for matching
            if word.endswith('ies'):
                return [word, word[:-3] + 'y']  # categories -> category
            elif word.endswith('es') and len(word) > 3:
                return [word, word[:-2]]  # boxes -> box, prices -> pric (but also catches 'es' suffix)
            elif word.endswith('s') and len(word) > 3:
                return [word, word[:-1]]  # prices -> price
            return [word]
        
        # Build normalized query words
        normalized_query = []
        for w in query_words:
            normalized_query.extend(normalize_word(w))
        
        # Method 1: Direct column name matching (exact or partial)
        for col in df_columns:
            col_lower = col.lower()
            col_clean = col_lower.replace('_', ' ').replace('-', ' ')
            
            # Check if column name or cleaned version appears in query
            if col_lower in query_lower or col_clean in query_lower:
                detected_column = col
                break
            
            # Check normalized versions
            col_normalized = normalize_word(col_lower)
            if any(cn in normalized_query for cn in col_normalized):
                detected_column = col
                break
            
            # Check if any significant word from column name appears in query
            col_words = [w for w in col_clean.split() if len(w) > 2]
            for col_word in col_words:
                col_word_normalized = normalize_word(col_word)
                if any(cw in normalized_query for cw in col_word_normalized):
                    detected_column = col
                    break
            if detected_column:
                break
        
        # Method 2: Try extracting phrases from query and matching to columns
        if not detected_column:
            words = query_text.lower().split()
            for i in range(len(words)):
                # Try progressively longer phrases (1, 2, 3 words)
                for j in range(i+1, min(i+4, len(words)+1)):
                    phrase = ' '.join(words[i:j])
                    found = find_column(phrase, df_columns)
                    if found:
                        detected_column = found
                        break
                if detected_column:
                    break
        
        return detected_intent, detected_column
    
    # ========== GREETINGS ==========
    if any(w in q for w in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
        return f"""👋 **Hello!  I'm DataGPT! ** Analyzing **{filename or 'your dataset'}** with **{df.shape[0]: ,} rows** × **{df.shape[1]} columns**. 

💡 **Try:** "what is this about? ", "show columns", "average {df.columns[0] if len(df.columns) > 0 else 'price'}"
""", None
    
    if any(w in q for w in ['thank', 'thanks']):
        return "😊 **You're welcome!** Happy to help!", None
    
    # ========== HELP ==========
    if any(w in q for w in ['help', 'what can you do', 'commands', 'capabilities']):
        sample_cols = list(df.columns[: 3])
        return f"""🤖 **DataGPT - I can answer 1000+ questions!**

**Dataset Info:**
- "What is this dataset about?"
- "Show all columns"
- "How many rows/columns?"
- "Summary/overview"

**Column Analysis (I have {df.shape[1]} columns!):**
- "What's the average {sample_cols[0] if sample_cols else 'price'}?"
- "Show me unique {sample_cols[1] if len(sample_cols) > 1 else 'categories'}"
- "Analyze {sample_cols[2] if len(sample_cols) > 2 else 'sales'}"
- "Max/min/median of [any column]"

**Data Quality:**
- "Show missing data"
- "Find duplicates"
- "Show correlations"

**Available columns:** {', '.join(df.columns[: 5])}{"..." if len(df.columns) > 5 else ""}

**Just ask naturally! ** 🚀
""", None
    
    # ========== DATASET OVERVIEW ==========
    if any(phrase in q for phrase in ['what is this', 'about this', 'dataset about', 'tell me about this', 'what data', 'describe this']):
        num_cols = len(df.select_dtypes(include=[np.number]).columns)
        cat_cols = len(df.select_dtypes(include=['object']).columns)
        total_cells = df.shape[0] * df. shape[1]
        complete = total_cells - df.isnull().sum().sum()
        quality = (complete / total_cells * 100) if total_cells > 0 else 0
        
        return f"""📁 **Dataset:  {filename or 'Your Data'}**

**Size:** {df.shape[0]:,} rows × {df.shape[1]} columns ({total_cells:,} cells)
**Memory:** {df.memory_usage(deep=True).sum()/1024**2:.2f} MB
**Quality Score:** {quality:.1f}%

**Column Types:**
- Numerical: {num_cols}
- Categorical: {cat_cols}
- DateTime: {len(df.select_dtypes(include=['datetime64']).columns)}

**Data Quality:**
- Missing:  {df.isnull().sum().sum():,} ({(df.isnull().sum().sum()/total_cells*100):.2f}%)
- Duplicates: {df.duplicated().sum():,}

**Columns:** {', '.join(df.columns[:5])}{"..." if len(df.columns) > 5 else ""}

💡 **Try:** "show columns", "summary", "average {df.columns[0]}"
""", None
    
    # ========== COLUMNS LISTING ==========
    if any(w in q for w in ['columns', 'column names', 'list columns', 'show columns', 'all columns', 'fields']):
        cols_info = []
        for i, col in enumerate(df.columns, 1):
            unique = df[col].nunique()
            missing = df[col].isnull().sum()
            insight = ""
            if missing > len(df) * 0.5:
                insight = " ⚠️ High missing"
            elif unique == 1:
                insight = " ⚠️ Constant"
            elif unique == len(df):
                insight = " 🔑 ID?"
            cols_info.append(f"{i}. **{col}** ({df[col].dtype}) - {unique:,} unique, {missing:,} missing{insight}")
        
        return f"""📋 **All {len(df. columns)} Columns:**

{chr(10).join(cols_info)}

💡 **Try:** "average {df.columns[0]}", "analyze {df.columns[1] if len(df.columns) > 1 else df.columns[0]}"
""", None
    
    # ========== SUMMARY ==========
    if any(w in q for w in ['summary', 'summarize', 'overview', 'describe data', 'statistics', 'stats']):
        num_cols = df.select_dtypes(include=[np.number]).columns
        summary = f"""📊 **Dataset Summary**

**Basic Info:**
- Rows: {df.shape[0]:,}
- Columns:  {df.shape[1]}
- Memory: {df.memory_usage(deep=True).sum()/1024**2:.2f} MB

**Data Quality:**
- Complete Rows: {df.dropna().shape[0]:,} ({df.dropna().shape[0]/df.shape[0]*100:. 1f}%)
- Missing Values: {df.isnull().sum().sum():,}
- Duplicates: {df. duplicated().sum():,}

**Column Types:**
- Numerical: {len(num_cols)}
- Categorical: {len(df.select_dtypes(include=['object']).columns)}
"""
        
        if len(num_cols) > 0:
            summary += f"\n**Numerical Summary:**\n"
            for col in num_cols[: 5]: 
                summary += f"\n**{col}:** Range {df[col].min():.2f} to {df[col].max():.2f}, Mean {df[col].mean():.2f}"
        
        return summary, None
    
    # ========== ROW/COLUMN COUNTS ==========
    if any(phrase in q for phrase in ['how many rows', 'number of rows', 'row count', 'total rows', 'rows in']):
        return f"📊 Dataset has **{df.shape[0]: ,} rows** (records).", None
    
    if any(phrase in q for phrase in ['how many columns', 'number of columns', 'column count', 'total columns']):
        return f"📋 Dataset has **{df.shape[1]} columns** (features).", None
    
    if any(w in q for w in ['size', 'how big', 'dimensions', 'shape']):
        return f"📐 **{df.shape[0]: ,} rows** × **{df.shape[1]} columns** = **{df.shape[0]*df.shape[1]: ,} cells**", None
    
    # ========== MISSING DATA ==========
    if any(w in q for w in ['missing', 'null', 'nan', 'empty']):
        total_missing = df.isnull().sum().sum()
        missing_by_col = df.isnull().sum()
        missing_cols = missing_by_col[missing_by_col > 0]. sort_values(ascending=False)
        
        response = f"""⚠️ **Missing Data Analysis**

**Total:** {total_missing:,} missing ({total_missing/(df.shape[0]*df. shape[1])*100:.2f}%)
**Columns Affected:** {len(missing_cols)} of {df.shape[1]}

"""
        if len(missing_cols) > 0:
            for col, count in missing_cols.head(10).items():
                severity = "🔴" if count/len(df) > 0.5 else "🟡" if count/len(df) > 0.2 else "🟢"
                response += f"{severity} **{col}**:  {count:,} ({count/len(df)*100:.1f}%)\n"
        else:
            response += "✅ **No missing data! **"
        
        return response, None
    
    # ========== DUPLICATES ==========
    if any(w in q for w in ['duplicate', 'duplicates', 'repeated']):
        dup = df.duplicated().sum()
        return f"""🔄 **Duplicates:** {dup:,} rows ({dup/df.shape[0]*100:.2f}%)
**Unique Rows:** {df.drop_duplicates().shape[0]:,}

{"✅ No duplicates!" if dup == 0 else "💡 Consider removing duplicates"}
""", None
    
    # ========== CORRELATIONS ==========
    if any(w in q for w in ['correlation', 'correlate', 'related', 'relationship']):
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) >= 2:
            corr = df[num_cols].corr()
            top_corr = []
            for i in range(len(corr. columns)):
                for j in range(i+1, len(corr.columns)):
                    top_corr.append((corr.columns[i], corr.columns[j], corr.iloc[i,j]))
            top_corr = sorted(top_corr, key=lambda x: abs(x[2]), reverse=True)[:10]
            
            response = "🔗 **Top 10 Correlations:**\n\n"
            for c1, c2, val in top_corr:
                strength = "💪 Strong" if abs(val) > 0.7 else "👌 Moderate" if abs(val) > 0.4 else "🤏 Weak"
                response += f"{strength}:  **{c1}** ↔️ **{c2}**:  {val:.3f}\n"
            return response, None
        return "⚠️ Need 2+ numerical columns", None
    
    # ========== SMART COLUMN-SPECIFIC QUERIES ==========
    intent, column = detect_intent_and_column(q, df.columns)
    
    if column:
        col = column
        
        # MEAN/AVERAGE
        if intent == 'mean' or any(w in q for w in ['mean', 'average', 'avg']):
            if df[col].dtype in ['float64', 'int64']: 
                return f"📊 **{col}** average: **{df[col].mean():.2f}** (median: {df[col].median():.2f}, min: {df[col].min():.2f}, max: {df[col].max():.2f})", None
            return f"⚠️ **{col}** is not numerical ({df[col].dtype}). Try 'show unique {col}'", None
        
        # MAX
        if intent == 'max' or any(w in q for w in ['max', 'maximum', 'highest', 'largest', 'biggest']):
            if df[col]. dtype in ['float64', 'int64']:
                max_val = df[col].max()
                min_val = df[col].min()
                return f"📈 **{col}** maximum: **{max_val:.2f}** (minimum: {min_val:.2f}, range: {max_val-min_val:.2f})", None
            return f"📈 **{col}** maximum: **{df[col].max()}**", None
        
        # MIN
        if intent == 'min' or any(w in q for w in ['min', 'minimum', 'lowest', 'smallest']):
            if df[col].dtype in ['float64', 'int64']:
                min_val = df[col].min()
                max_val = df[col].max()
                return f"📉 **{col}** minimum: **{min_val:.2f}** (maximum: {max_val:.2f}, range: {max_val-min_val:.2f})", None
            return f"📉 **{col}** minimum: **{df[col].min()}**", None
        
        # MEDIAN
        if intent == 'median' or 'median' in q: 
            if df[col].dtype in ['float64', 'int64']:
                return f"📊 **{col}** median: **{df[col].median():.2f}** (mean: {df[col].mean():.2f})", None
            return f"⚠️ Median only for numerical data", None
        
        # SUM/TOTAL
        if intent == 'sum' or any(w in q for w in ['sum', 'total']):
            if df[col]. dtype in ['float64', 'int64']:
                return f"➕ **{col}** total: **{df[col].sum():,.2f}** (average: {df[col].mean():.2f})", None
            return f"⚠️ Sum only for numerical data", None
        
        # COUNT
        if intent == 'count' or any(phrase in q for phrase in ['how many', 'count', 'number of']):
            count = len(df[col]. dropna())
            return f"🔢 **{col}** has **{count:,} values** ({df[col].nunique():,} unique)", None
        
        # UNIQUE/DISTINCT
        if intent == 'unique' or intent == 'show' or any(w in q for w in ['unique', 'distinct', 'different', 'show', 'list']):
            unique = df[col].nunique()
            top_5 = df[col].value_counts().head(10)
            response = f"🔢 **{col}** has **{unique:,} unique values**\n\n**Top 10:**\n"
            for val, count in top_5.items():
                response += f"- **{val}**:  {count:,} ({count/len(df)*100:.1f}%)\n"
            return response, None
        
        # STD/VARIANCE
        if intent == 'std' or any(w in q for w in ['std', 'standard deviation', 'variance']):
            if df[col]. dtype in ['float64', 'int64']:
                return f"📊 **{col}** std dev: **{df[col].std():.2f}** (variance: {df[col].var():.2f})", None
            return f"⚠️ Standard deviation only for numerical data", None
        
        # MODE
        if intent == 'mode' or any(phrase in q for phrase in ['mode', 'most common', 'most frequent']):
            mode_val = df[col].mode()[0] if not df[col].mode().empty else 'N/A'
            mode_count = (df[col] == mode_val).sum() if mode_val != 'N/A' else 0
            return f"📊 **{col}** most common: **{mode_val}** (appears {mode_count:,} times, {mode_count/len(df)*100:.1f}%)", None
        
        # ANALYZE - Full analysis
        if intent == 'analyze' or any(w in q for w in ['analyze', 'analysis', 'detailed', 'full', 'tell me about', 'describe', 'info']):
            if df[col].dtype in ['float64', 'int64']:
                stats = f"""🔍 **Complete Analysis:  {col}**

**Statistics:**
- Mean: {df[col].mean():.2f}
- Median: {df[col].median():.2f}
- Std Dev: {df[col]. std():.2f}
- Min: {df[col].min():.2f}
- Max: {df[col].max():.2f}
- Range: {df[col].max() - df[col].min():.2f}

**Quality:**
- Missing: {df[col].isnull().sum():,} ({df[col].isnull().sum()/len(df)*100:.2f}%)
- Unique: {df[col].nunique():,}
- Zeros: {(df[col] == 0).sum():,}

**Insights:**
"""
                # Trend
                if len(df) > 1:
                    corr = df[col].corr(pd.Series(range(len(df))))
                    if corr > 0.3:
                        stats += "- 📈 Upward trend\n"
                    elif corr < -0.3:
                        stats += "- 📉 Downward trend\n"
                    else: 
                        stats += "- ➡️ No clear trend\n"
                
                # Variability
                cv = (df[col]. std() / df[col].mean() * 100) if df[col].mean() != 0 else 0
                if cv < 15:
                    stats += "- 🟢 Low variability\n"
                elif cv < 30:
                    stats += "- 🟡 Moderate variability\n"
                else:
                    stats += "- 🔴 High variability\n"
                
                anom_fig, _ = perform_anomaly_detection(df, col)
                fore_fig, _ = perform_forecasting(df, col)
                figs = []
                if anom_fig:  figs.append(anom_fig)
                if fore_fig: figs.append(fore_fig)
                
                return stats, figs if figs else None
            else:
                vc = df[col].value_counts()
                response = f"""🔍 **Analysis:  {col}**

**Type:** {df[col].dtype}
**Unique:** {df[col].nunique():,}
**Missing:** {df[col].isnull().sum():,} ({df[col].isnull().sum()/len(df)*100:.2f}%)
**Most Common:** {df[col].mode()[0] if not df[col].mode().empty else 'N/A'}

**Top 10 Values:**
"""
                for val, count in vc.head(10).items():
                    response += f"- **{val}**: {count: ,} ({count/len(df)*100:.1f}%)\n"
                return response, None
        
        # DEFAULT: Column found but no specific intent - provide quick summary
        if df[col].dtype in ['float64', 'int64']:
            col_data = df[col].dropna()
            if len(col_data) == 0:
                return f"""📊 **{col}** Quick Summary:
- Type: Numerical
- All values are missing (NaN)

💡 Try: "show missing data" for more details!
""", None
            return f"""📊 **{col}** Quick Summary:
- Mean: **{col_data.mean():.2f}**
- Median: {col_data.median():.2f}
- Min: {col_data.min():.2f}
- Max: {col_data.max():.2f}
- Unique values: {col_data.nunique():,}

💡 Try: "average {col}", "analyze {col}", "show unique {col}" for more details!
""", None
        else:
            top_3 = df[col].value_counts().head(3)
            if len(top_3) == 0:
                top_vals = "No values available"
            else:
                top_vals = ', '.join([f"**{v}** ({c:,})" for v, c in top_3.items()])
            return f"""📊 **{col}** Quick Summary:
- Type: {df[col].dtype}
- Unique values: **{df[col].nunique():,}**
- Top values: {top_vals}

💡 Try: "analyze {col}", "show unique {col}" for more details!
""", None
    
    # ========== FALLBACK WITH SMART SUGGESTIONS ==========
    # Try to extract potential column names from query
    potential_columns = []
    words = q.split()
    for col in df.columns:
        col_words = col.lower().replace('_', ' ').replace('-', ' ').split()
        if any(word in words for word in col_words if len(word) > 2):
            potential_columns.append(col)
    
    if potential_columns:
        suggestions = ', '.join([f'"{col}"' for col in potential_columns[: 3]])
        return f"""💡 **I found these related columns:** {suggestions}

**Try asking:**
- "What's the average {potential_columns[0]}?"
- "Show unique {potential_columns[0]}"
- "Analyze {potential_columns[0]}"

**Or type "help" to see all I can do!** 🤔
""", None
    
    # Final fallback
    return f"""💡 **I can help! **

**Available columns:** {', '.join(df. columns[:5])}{"..." if len(df.columns) > 5 else ""}

**Try:**
- "What is this dataset about?"
- "Show columns" ({df.shape[1]} available)
- "Average {df.columns[0]}"
- "Analyze {df.columns[0]}"

Type **"help"** for all capabilities!  🤔
""", None

# =============================================================================
# LAYOUT COMPONENTS
# =============================================================================

def create_navbar():
    return dbc.Navbar(
        dbc.Container([
            dbc.Row([dbc.Col([html.Div([html.Div("🤖", style={'fontSize': '1.8rem', 'marginRight': '10px'}), html.Span("AI Data Analyst", className="brand-logo")], style={'display': 'flex', 'alignItems': 'center'})], width="auto")], align="center", className="g-0 w-100"),
            dbc.Row([dbc.Col([dbc.Nav([
                dbc.NavItem(dbc.NavLink([html.Span("🏠 "), "Dashboard"], href="/", className="nav-link")),
                dbc.NavItem(dbc.NavLink([html.Span("🧹 "), "Cleaning"], href="/cleaning", className="nav-link")),
                dbc.NavItem(dbc.NavLink([html.Span("📊 "), "Visualization"], href="/visualization", className="nav-link")),
                dbc.NavItem(dbc.NavLink([html.Span("🧠 "), "AutoML"], href="/automl", className="nav-link")),
                dbc.NavItem(dbc.NavLink([html.Span("💬 "), "DataGPT"], href="/datagpt", className="nav-link")),
                dbc.NavItem(dbc.NavLink([html.Span("📥 "), "Export"], href="/export", className="nav-link")),
            ], navbar=True, className="ms-auto")])], className="g-0 w-100 justify-content-end"),
        ], fluid=True),
        className="custom-nav",
        dark=True,
    )

def create_upload_section():
    return html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                html.Div("☁️", style={'fontSize': '7rem', 'marginBottom': '20px', 'filter': 'drop-shadow(0 0 20px rgba(255,255,255,0.6))'}),
                html.Div("⬆️", style={'fontSize': '4rem', 'marginTop': '-60px', 'marginBottom': '30px'}),
                html.H4("Upload Your Dataset", style={'color': '#ffffff', 'marginBottom': '15px', 'fontWeight': '700', 'fontSize': '1.8rem'}),
                html.P(["Drag and drop your CSV file here or ", html. Span("click to browse", style={'color': '#ffd93d', 'fontWeight': '700', 'cursor': 'pointer', 'textDecoration': 'underline'})], style={'color': '#ffffff', 'fontSize': '1.1rem', 'marginBottom': '15px'}),
                html.Small("📄 Supported format: CSV", style={'color': 'rgba(255, 255, 255, 0.9)', 'fontSize': '0.95rem'})
            ], className="upload-zone"),
            multiple=False
        ),
        html.Div(id='upload-status', className='mt-4')
    ], className="mb-5")

def create_dashboard_layout():
    return html.Div([
        html.H2("📊 Dashboard Overview", className="section-title mb-4"),
        dbc.Row([
            dbc.Col([html.Div([html.Div("📊", className="kpi-icon"), html.Div(id="kpi-rows", className="kpi-value"), html.Div("Total Rows", className="kpi-label")], className="kpi-card")], md=3, className="mb-4"),
            dbc.Col([html. Div([html.Div("📋", className="kpi-icon"), html.Div(id="kpi-features", className="kpi-value"), html.Div("Features", className="kpi-label")], className="kpi-card")], md=3, className="mb-4"),
            dbc.Col([html.Div([html.Div("⚠️", className="kpi-icon"), html.Div(id="kpi-missing", className="kpi-value"), html.Div("Missing Values", className="kpi-label")], className="kpi-card")], md=3, className="mb-4"),
            dbc.Col([html.Div([html.Div("🔄", className="kpi-icon"), html.Div(id="kpi-duplicates", className="kpi-value"), html.Div("Duplicates", className="kpi-label")], className="kpi-card")], md=3, className="mb-4"),
        ]),
        html.Div([html.H4("📄 Data Preview", style={'color': '#2c3e50', 'marginBottom': '20px'}), html.Div(id="data-preview", className="table-container")], className="glass-card p-4")
    ])

def create_cleaning_layout():
    return html.Div([
        html.H2("🧹 Data Cleaning", className="section-title mb-4"),
        dbc.Row([dbc.Col([html.Div([
            html.H4("🧹 Auto-Clean Pipeline", style={'color': '#2c3e50'}),
            html.P("Automatically remove duplicates, handle missing values", style={'color': '#666', 'marginBottom': '25px'}),
            dbc.Button([html.Span("✨ "), "Clean Data Now"], id="btn-clean", size="lg", className="btn-custom w-100"),
            html.Div(id="cleaning-status", className="mt-4")
        ], className="glass-card p-4 mb-4")])]),
        dbc.Row([
            dbc.Col([html.Div([html.H5("Before", style={'color': '#2c3e50'}), dcc.Graph(id="health-before", style={'height': '300px'})], className="glass-card p-4")], md=6),
            dbc.Col([html.Div([html.H5("After", style={'color': '#2c3e50'}), dcc.Graph(id="health-after", style={'height': '300px'})], className="glass-card p-4")], md=6)
        ])
    ])

def create_visualization_layout():
    return html.Div([
        html.H2("📊 Data Visualization", className="section-title mb-4"),
        dbc.Row([dbc.Col([html.Div([html.H5("📊 Distribution", style={'color': '#2c3e50'}), dcc.Dropdown(id="pie-column-selector", className="mb-3", searchable=False, clearable=False, placeholder="Select a column..."), dcc.Graph(id="pie-chart", style={'height': '400px'})], className="glass-card p-4 mb-4")])]),
        dbc.Row([dbc.Col([html.Div([html.H5("🔥 Correlation", style={'color': '#2c3e50'}), dcc.Graph(id="correlation-heatmap", style={'height': '500px'})], className="glass-card p-4 mb-4")])]),
        dbc.Row([dbc.Col([html.Div([html.H5("📈 Distribution", style={'color': '#2c3e50'}), dcc.Dropdown(id="dist-column-selector", className="mb-3", searchable=False, clearable=False, placeholder="Select a column..."), dcc.Graph(id="distribution-plot", style={'height': '500px'})], className="glass-card p-4")])])
    ])

def create_automl_layout():
    return html.Div([
        html.H2("🧠 AutoML Agent", className="section-title mb-4"),
        dbc.Row([dbc.Col([html.Div([
            html.H4("🤖 Automated ML", style={'color': '#2c3e50'}),
            html.P("Select target and train model", style={'color': '#666'}),
            dcc.Dropdown(id="target-column-selector", className="mb-3", searchable=False, clearable=False, placeholder="Select target column..."),
            dbc.Button([html.Span("🚀 "), "Train Model"], id="btn-train", size="lg", className="btn-custom w-100"),
            html.Div(id="training-status", className="mt-4")
        ], className="glass-card p-4 mb-4")])]),
        html.Div(id="automl-results-container", children=[
            dbc.Row([dbc.Col([html.Div([html.Div(id="model-results")], className="glass-card p-4")], className="mb-4")]),
            dbc.Row([dbc.Col([html.Div([dcc.Graph(id="feature-importance-chart", config={'displayModeBar': True, 'scrollZoom': False}, style={'height': '500px'})], className="glass-card p-4")])])
        ], style={'display': 'none'})
    ])

def create_datagpt_layout():
    return html.Div([
        html.H2("💬 DataGPT - Answers 1000+ Questions!", className="section-title mb-4"),
        dbc.Row([dbc.Col([html.Div([
            html.Div([
                html.Div(id="chat-history", className="chat-container", style={'minHeight': '450px', 'marginBottom': '20px'}),
                dbc.InputGroup([
                    dbc.Input(id="chat-input", placeholder="Ask ANYTHING!  Try: 'what is this about?', 'show columns', 'analyze [column]'", type="text"),
                    dbc.Button([html.Span("📤")], id="btn-send-chat", className="btn-custom")
                ])
            ])
        ], className="glass-card p-4")])]),
        html.Div(id="chat-graphs")
    ])

def create_export_layout():
    return html.Div([
        html.H2("📥 Export Professional Report", className="section-title mb-4"),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div([
                        html.Div("📊", style={'fontSize': '4rem', 'marginBottom': '20px', 'textAlign': 'center'}),
                        html.H4("Export Cleaned Dataset", style={'color': '#2c3e50', 'textAlign': 'center'}),
                        html.P("Download your cleaned data as a CSV file", style={'color': '#666', 'textAlign': 'center', 'marginBottom': '25px'}),
                        dbc.Button([html.Span("📥 "), "Download CSV"], id="btn-export-csv", size="lg", className="btn-custom w-100"),
                        html.Div(id="export-csv-status", className="mt-3")
                    ], className="glass-card p-4", style={'height': '100%'})
                ])
            ], md=6, className="mb-4"),
            dbc.Col([
                html.Div([
                    html.Div([
                        html.Div("📑", style={'fontSize': '4rem', 'marginBottom': '20px', 'textAlign': 'center'}),
                        html.H4("Export Professional Report", style={'color': '#2c3e50', 'textAlign': 'center'}),
                        html.P("Generate a comprehensive HTML report with charts and analysis", style={'color': '#666', 'textAlign': 'center', 'marginBottom': '25px'}),
                        dbc.Button([html.Span("📑 "), "Generate Report"], id="btn-export-report", size="lg", className="btn-custom w-100"),
                        html.Div(id="export-report-status", className="mt-3")
                    ], className="glass-card p-4", style={'height': '100%'})
                ])
            ], md=6, className="mb-4")
        ]),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4("📋 Export Preview", style={'color': '#2c3e50', 'marginBottom': '20px'}),
                    html.Div(id="export-preview", children=[
                        html.Div([
                            html.Div("👆", style={'fontSize': '3rem', 'marginBottom': '15px'}),
                            html.P("Click on an export button above to preview and download your data", style={'color': '#666', 'textAlign': 'center'})
                        ], style={'textAlign': 'center', 'padding': '50px'})
                    ])
                ], className="glass-card p-4")
            ])
        ]),
        dcc.Download(id="download-csv"),
        dcc.Download(id="download-report")
    ])

# =============================================================================
# MAIN LAYOUT
# =============================================================================

app. layout = html.Div([
    dcc.Store(id='session-id', data=str(uuid.uuid4())),
    dcc.Store(id='data-loaded', data=False),
    dcc.Location(id='url', refresh=False),
    create_navbar(),
    dbc.Container([
        html.Div(id='upload-section-container', children=[create_upload_section()], className="mt-5"),
        html.Div(id='page-content', className="main-container")
    ], fluid=True)
])

# =============================================================================
# CALLBACKS
# =============================================================================

@app. callback(
    [Output('upload-status', 'children'), Output('session-id', 'data'), Output('data-loaded', 'data')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename'), State('session-id', 'data')]
)
def upload_file(contents, filename, session_id):
    if contents is None:
        raise PreventUpdate
    df = parse_upload_contents(contents, filename)
    if df is None:
        return dbc.Alert([html. Span("❌ "), "Error"], color="danger"), session_id, False
    SERVER_DATA_CACHE[session_id] = {'original':  df, 'cleaned': None, 'filename': filename}
    return dbc.Alert([html. Span("✅ "), f"Uploaded:  {filename} ({df.shape[0]: ,} rows)"], color="success"), session_id, True

@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname'), Input('data-loaded', 'data')],
    [State('session-id', 'data')]
)
def display_page(pathname, data_loaded, session_id):
    if not data_loaded or session_id not in SERVER_DATA_CACHE:
        return html.Div([html.Div("⚠️", style={'fontSize': '5rem'}), html.H3("Please upload dataset", style={'color': '#2c3e50'})], style={'textAlign': 'center', 'padding': '100px'})
    if pathname == '/cleaning':  return create_cleaning_layout()
    elif pathname == '/visualization': return create_visualization_layout()
    elif pathname == '/automl': return create_automl_layout()
    elif pathname == '/datagpt': return create_datagpt_layout()
    elif pathname == '/export': return create_export_layout()
    else: return create_dashboard_layout()

@app.callback(
    [Output('kpi-rows', 'children'), Output('kpi-features', 'children'), Output('kpi-missing', 'children'), Output('kpi-duplicates', 'children'), Output('data-preview', 'children')],
    [Input('data-loaded', 'data')],
    [State('session-id', 'data')]
)
def update_dashboard(data_loaded, session_id):
    if not data_loaded or session_id not in SERVER_DATA_CACHE:
        raise PreventUpdate
    df = get_dataframe(session_id)
    stats = get_health_stats(df)
    table = dash_table.DataTable(
        data=df.head(10).to_dict('records'),
        columns=[{'name': c, 'id': c} for c in df.columns],
        style_cell={'textAlign': 'left', 'padding': '12px', 'backgroundColor': 'white', 'color': '#333'},
        style_header={'backgroundColor': '#667eea', 'color': 'white'}
    )
    return f"{df.shape[0]:,}", str(df.shape[1]), f"{stats['missing']:,}", f"{stats['duplicates']:,}", table

@app.callback(
    [Output('cleaning-status', 'children'), Output('health-before', 'figure'), Output('health-after', 'figure')],
    [Input('btn-clean', 'n_clicks')],
    [State('session-id', 'data')]
)
def clean_data(n, sid):
    if n is None or sid not in SERVER_DATA_CACHE:  raise PreventUpdate
    df_orig = SERVER_DATA_CACHE[sid]['original']
    df_clean = auto_clean_data(df_orig)
    SERVER_DATA_CACHE[sid]['cleaned'] = df_clean
    sb = get_health_stats(df_orig)
    sa = get_health_stats(df_clean)
    fb = go.Figure(data=[go.Bar(x=['Missing', 'Dups'], y=[sb['missing'], sb['duplicates']], marker=dict(color=['#ff6b9d', '#ffd93d']))])
    fb.update_layout(title="Before", template='plotly_white', height=300)
    fa = go.Figure(data=[go.Bar(x=['Missing', 'Dups'], y=[sa['missing'], sa['duplicates']], marker=dict(color=['#00d9a5', '#667eea']))])
    fa.update_layout(title="After", template='plotly_white', height=300)
    return dbc.Alert([html.Span("✅ "), "Done! "], color="success"), fb, fa

@app.callback(
    [Output('pie-column-selector', 'options'), Output('pie-column-selector', 'value')],
    [Input('url', 'pathname'), Input('data-loaded', 'data')],
    [State('session-id', 'data')]
)
def update_pie_sel(p, d, sid):
    if p != '/visualization' or not d or sid not in SERVER_DATA_CACHE:  raise PreventUpdate
    df = get_dataframe(sid)
    if df is None or len(df.columns) == 0:
        return [], None
    return [{'label': c, 'value': c} for c in df.columns], df.columns[0]

@app.callback(
    Output('pie-chart', 'figure'),
    [Input('pie-column-selector', 'value')],
    [State('session-id', 'data')]
)
def update_pie(c, sid):
    if not c or sid not in SERVER_DATA_CACHE: raise PreventUpdate
    df = get_dataframe(sid)
    return create_smart_pie_chart(df, c)

@app.callback(
    Output('correlation-heatmap', 'figure'),
    [Input('url', 'pathname'), Input('data-loaded', 'data')],
    [State('session-id', 'data')]
)
def update_corr(p, d, sid):
    if p != '/visualization' or not d or sid not in SERVER_DATA_CACHE: raise PreventUpdate
    df = get_dataframe(sid)
    return create_correlation_heatmap(df)

@app.callback(
    [Output('dist-column-selector', 'options'), Output('dist-column-selector', 'value')],
    [Input('url', 'pathname'), Input('data-loaded', 'data')],
    [State('session-id', 'data')]
)
def update_dist_sel(p, d, sid):
    if p != '/visualization' or not d or sid not in SERVER_DATA_CACHE: raise PreventUpdate
    df = get_dataframe(sid)
    if df is None or len(df.columns) == 0:
        return [], None
    nc = df.select_dtypes(include=[np.number]).columns.tolist()
    return [{'label': c, 'value':  c} for c in nc], nc[0] if nc else None

@app. callback(
    Output('distribution-plot', 'figure'),
    [Input('dist-column-selector', 'value')],
    [State('session-id', 'data')]
)
def update_dist(c, sid):
    if not c or sid not in SERVER_DATA_CACHE: raise PreventUpdate
    df = get_dataframe(sid)
    return create_distribution_plot(df, c)

@app.callback(
    [Output('target-column-selector', 'options'), Output('target-column-selector', 'value')],
    [Input('url', 'pathname'), Input('data-loaded', 'data')],
    [State('session-id', 'data')]
)
def update_target_sel(p, d, sid):
    if p != '/automl' or not d or sid not in SERVER_DATA_CACHE:  raise PreventUpdate
    df = get_dataframe(sid)
    if df is None or len(df.columns) == 0:
        return [], None
    return [{'label': c, 'value':  c} for c in df.columns], df.columns[0]

@app.callback(
    [Output('training-status', 'children'), Output('model-results', 'children'), Output('feature-importance-chart', 'figure'), Output('automl-results-container', 'style')],
    [Input('btn-train', 'n_clicks')],
    [State('target-column-selector', 'value'), State('session-id', 'data')]
)
def train_model_cb(n, tc, sid):
    if n is None or not tc or sid not in SERVER_DATA_CACHE: raise PreventUpdate
    df = get_dataframe(sid)
    try:
        r = train_ml_model(df, tc)
        
        # Build model comparison table if multiple models were tested
        model_comparison = []
        if 'all_results' in r and len(r['all_results']) > 1:
            for res in sorted(r['all_results'], key=lambda x: x['score'], reverse=True):
                is_best = res['model'] == r.get('best_model_name', '')
                model_comparison.append(
                    html.Tr([
                        html.Td([html.Strong(res['model']) if is_best else res['model'], html.Span(" ⭐", style={'color': '#ffd93d'}) if is_best else ""], style={'color': '#667eea' if is_best else '#333'}),
                        html.Td(f"{res['score']:.4f}", style={'fontWeight': '700' if is_best else '400', 'color': '#00b894' if is_best else '#333'})
                    ])
                )
        
        results_content = [
            html.H5(f"🎯 {r['task_type'].title()}", style={'color': '#2c3e50', 'marginBottom': '15px'}),
            html.Div([
                html.Span(f"Best Model: ", style={'color': '#666'}),
                html.Span(f"{r.get('best_model_name', 'Random Forest')}", style={'fontSize': '1.2rem', 'color': '#764ba2', 'fontWeight': '600'})
            ], style={'marginBottom': '10px'}),
            html.Div([
                html.Span(f"{r['metric_name']}: "),
                html.Span(f"{r['score']:.4f}", style={'fontSize': '2rem', 'color': '#667eea', 'fontWeight': '700'})
            ])
        ]
        
        # Add model comparison table if available
        if model_comparison:
            results_content.extend([
                html.Hr(style={'margin': '20px 0'}),
                html.H6("📊 Model Comparison", style={'color': '#2c3e50', 'marginBottom': '10px'}),
                html.Table([
                    html.Thead(html.Tr([html.Th("Model"), html.Th(r['metric_name'])], style={'backgroundColor': '#f8f9ff'})),
                    html.Tbody(model_comparison)
                ], style={'width': '100%', 'borderCollapse': 'collapse', 'fontSize': '0.9rem'})
            ])
        
        return (
            dbc.Alert([html.Span("✅ "), f"Training Complete! Best: {r.get('best_model_name', 'Model')}"], color="success"),
            html.Div(results_content),
            create_feature_importance_chart(r['feature_importance']),
            {'display': 'block'}
        )
    except Exception as e:
        return dbc.Alert([html.Span("❌ "), str(e)], color="danger"), html.Div(), go.Figure(), {'display': 'none'}


@app.callback(
    [Output('chat-history', 'children'), Output('chat-graphs', 'children'), Output('chat-input', 'value')],
    [Input('btn-send-chat', 'n_clicks')],
    [State('chat-input', 'value'), State('chat-history', 'children'), State('session-id', 'data')]
)
def chat(n, ui, ch, sid):
    if n is None or not ui or sid not in SERVER_DATA_CACHE: raise PreventUpdate
    df = get_dataframe(sid)
    fn = SERVER_DATA_CACHE. get(sid, {}).get('filename', '')
    rt, figs = parse_datagpt_query(ui, df, fn)
    if ch is None:  ch = []
    ch.append(html.Div([html.Strong("👤 You", style={'color': '#667eea', 'display': 'block', 'marginBottom': '5px'}), html.Div(ui, style={'color': 'white'})], className="chat-message-user"))
    ch.append(html.Div([html.Strong("🤖 DataGPT", style={'color': '#764ba2', 'display': 'block', 'marginBottom':  '5px'}), dcc. Markdown(rt, style={'color': '#333'})], className="chat-message-bot"))
    gc = []
    if figs and isinstance(figs, list):
        for f in figs:
            if f:  gc.append(dbc.Col([html.Div([dcc.Graph(figure=f)], className="glass-card p-3")], md=6))
        if gc: 
            gc.append(dbc.Col([html.Div([dcc.Graph(figure=create_sankey_diagram(df))], className="glass-card p-3")], md=6))
        graphs = dbc.Row(gc, className="mt-4")
    else:
        graphs = html.Div()
    return ch, graphs, ""

# Export CSV callback
@app.callback(
    [Output('download-csv', 'data'), Output('export-csv-status', 'children'), Output('export-preview', 'children', allow_duplicate=True)],
    [Input('btn-export-csv', 'n_clicks')],
    [State('session-id', 'data')],
    prevent_initial_call=True
)
def export_csv(n, sid):
    if n is None or sid not in SERVER_DATA_CACHE:
        raise PreventUpdate
    
    df = get_dataframe(sid)
    filename = SERVER_DATA_CACHE.get(sid, {}).get('filename', 'data.csv')
    
    # Create preview
    preview = html.Div([
        html.H5("✅ CSV Export Ready", style={'color': '#00b894', 'marginBottom': '15px'}),
        html.P(f"Dataset: {filename}", style={'color': '#666'}),
        html.P(f"Rows: {len(df):,} | Columns: {len(df.columns)}", style={'color': '#666', 'marginBottom': '15px'}),
        html.Div([
            dash_table.DataTable(
                data=df.head(5).to_dict('records'),
                columns=[{'name': c, 'id': c} for c in df.columns],
                style_cell={'textAlign': 'left', 'padding': '10px', 'backgroundColor': 'white', 'color': '#333', 'fontSize': '0.85rem'},
                style_header={'backgroundColor': '#667eea', 'color': 'white', 'fontWeight': '600'}
            )
        ], style={'overflowX': 'auto'})
    ])
    
    # Export file
    export_filename = filename.replace('.csv', '_cleaned.csv') if '.csv' in filename else f"{filename}_cleaned.csv"
    
    return dcc.send_data_frame(df.to_csv, export_filename, index=False), dbc.Alert([html.Span("✅ "), "CSV downloaded!"], color="success"), preview

# Export Report callback
@app.callback(
    [Output('download-report', 'data'), Output('export-report-status', 'children'), Output('export-preview', 'children')],
    [Input('btn-export-report', 'n_clicks')],
    [State('session-id', 'data')],
    prevent_initial_call=True
)
def export_report(n, sid):
    if n is None or sid not in SERVER_DATA_CACHE:
        raise PreventUpdate
    
    df = get_dataframe(sid)
    filename = SERVER_DATA_CACHE.get(sid, {}).get('filename', 'Dataset')
    
    # Generate professional report
    html_report = generate_professional_report(df, filename)
    
    # Create preview
    stats = get_health_stats(df)
    completeness = ((df.shape[0] * df.shape[1] - stats['missing']) / (df.shape[0] * df.shape[1]) * 100) if df.shape[0] * df.shape[1] > 0 else 0
    
    preview = html.Div([
        html.H5("✅ Professional Report Generated", style={'color': '#00b894', 'marginBottom': '20px'}),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div("📊", style={'fontSize': '2rem', 'marginBottom': '10px'}),
                    html.H6(f"{len(df):,}", style={'color': '#667eea', 'fontSize': '1.5rem', 'fontWeight': '700'}),
                    html.P("Records", style={'color': '#666', 'fontSize': '0.85rem'})
                ], style={'textAlign': 'center', 'padding': '15px', 'background': '#f8f9ff', 'borderRadius': '10px'})
            ], md=3),
            dbc.Col([
                html.Div([
                    html.Div("📋", style={'fontSize': '2rem', 'marginBottom': '10px'}),
                    html.H6(f"{len(df.columns)}", style={'color': '#667eea', 'fontSize': '1.5rem', 'fontWeight': '700'}),
                    html.P("Features", style={'color': '#666', 'fontSize': '0.85rem'})
                ], style={'textAlign': 'center', 'padding': '15px', 'background': '#f8f9ff', 'borderRadius': '10px'})
            ], md=3),
            dbc.Col([
                html.Div([
                    html.Div("✅", style={'fontSize': '2rem', 'marginBottom': '10px'}),
                    html.H6(f"{completeness:.1f}%", style={'color': '#00b894', 'fontSize': '1.5rem', 'fontWeight': '700'}),
                    html.P("Quality", style={'color': '#666', 'fontSize': '0.85rem'})
                ], style={'textAlign': 'center', 'padding': '15px', 'background': '#f0fff4', 'borderRadius': '10px'})
            ], md=3),
            dbc.Col([
                html.Div([
                    html.Div("📈", style={'fontSize': '2rem', 'marginBottom': '10px'}),
                    html.H6(f"{len(df.select_dtypes(include=[np.number]).columns)}", style={'color': '#764ba2', 'fontSize': '1.5rem', 'fontWeight': '700'}),
                    html.P("Numeric Cols", style={'color': '#666', 'fontSize': '0.85rem'})
                ], style={'textAlign': 'center', 'padding': '15px', 'background': '#faf5ff', 'borderRadius': '10px'})
            ], md=3),
        ]),
        html.Div([
            html.P("📑 Your professional HTML report includes:", style={'color': '#2c3e50', 'fontWeight': '600', 'marginTop': '20px', 'marginBottom': '10px'}),
            html.Ul([
                html.Li("Executive summary with key metrics"),
                html.Li("Data quality analysis"),
                html.Li("Interactive visualizations (pie charts, histograms, correlation heatmap)"),
                html.Li("Detailed column analysis"),
                html.Li("Professional styling ready for presentation")
            ], style={'color': '#666', 'paddingLeft': '20px'})
        ])
    ])
    
    # Export file
    report_filename = filename.replace('.csv', '_report.html') if '.csv' in filename else f"{filename}_report.html"
    
    return dict(content=html_report, filename=report_filename, type='text/html'), dbc.Alert([html.Span("✅ "), "Report downloaded!"], color="success"), preview

# =============================================================================
# RUN APPLICATION
# =============================================================================

if __name__ == '__main__':  
    print("="*70)
    print("🚀 AI Data Analyst with ULTRA-POWERED DataGPT!")
    print("="*70)
    print("\n📍 Open:  http://localhost:8050")
    print("\n💬 DataGPT handles 1000+ question variations!")
    print("\n💡 Press CTRL+C to stop\n")
    print("="*70)
    app.run(debug=False, host='0.0.0.0', port=8050)