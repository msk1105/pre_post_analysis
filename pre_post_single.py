import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO

# Page config
st.set_page_config(page_title="Pre-Post Analysis Tool", page_icon="📊", layout="wide")

# Title and instructions
st.title("📊 Pre-Post Comparison Tool")
st.markdown("""
**Instructions:**
1. Upload your Excel or CSV file
2. Select the PRE and POST columns
3. Click 'Analyze' to see results (automatic normality check included)
4. Download the report

*The tool automatically checks normality and chooses the appropriate test:*
- ✅ **Normal data** → Paired t-test
- ⚠️ **Non-normal data** → Wilcoxon signed-rank test
""")

# Sidebar for file upload
with st.sidebar:
    st.header("📁 Upload Data")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['xlsx', 'xls', 'csv'],
        help="Upload Excel or CSV file with pre and post scores"
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01)
    normality_test = st.selectbox(
        "Normality test",
        ["Shapiro-Wilk (recommended)", "Kolmogorov-Smirnov"],
        help="Shapiro-Wilk is better for small samples (n<50)"
    )
    
    st.markdown("---")
    st.markdown("### 📖 Data Format Example")
    example_df = pd.DataFrame({
        'ID': [1, 2, 3, 4, 5],
        'Pre_Score': [65, 72, 58, 81, 69],
        'Post_Score': [78, 85, 71, 89, 82]
    })
    st.dataframe(example_df, use_container_width=True)

# Main content
if uploaded_file is not None:
    try:
        # Read file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success(f"✅ File loaded successfully! ({len(df)} rows)")
        
        # Show data preview
        with st.expander("👀 Preview Your Data", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)
        
        # Column selection
        col1, col2 = st.columns(2)
        with col1:
            pre_col = st.selectbox("📉 Select PRE column:", df.columns, index=0)
        with col2:
            post_col = st.selectbox("📈 Select POST column:", df.columns, index=min(1, len(df.columns)-1))
        
        # Analyze button
        if st.button("🔬 Analyze Data", type="primary", use_container_width=True):
            
            # Extract and clean data
            valid_indices = df[[pre_col, post_col]].dropna().index
            pre_data = pd.to_numeric(df.loc[valid_indices, pre_col], errors='coerce')
            post_data = pd.to_numeric(df.loc[valid_indices, post_col], errors='coerce')
            
            # Remove any remaining NaN
            valid_mask = ~(pre_data.isna() | post_data.isna())
            pre_data = pre_data[valid_mask]
            post_data = post_data[valid_mask]
            
            n = len(pre_data)
            
            if n < 3:
                st.error("❌ Not enough valid data pairs. Need at least 3 complete pairs for statistical testing.")
            else:
                # Calculate differences
                difference = post_data - pre_data
                
                # === NORMALITY TESTS ===
                st.markdown("---")
                st.header("🔍 Normality Assessment")
                
                # Perform normality tests
                if normality_test == "Shapiro-Wilk (recommended)":
                    stat_diff, p_diff = stats.shapiro(difference)
                    test_name = "Shapiro-Wilk"
                else:
                    stat_diff, p_diff = stats.kstest(difference, 'norm', args=(difference.mean(), difference.std()))
                    test_name = "Kolmogorov-Smirnov"
                
                # Check normality
                is_normal = p_diff > alpha
                
                # Display normality results
                norm_col1, norm_col2, norm_col3 = st.columns(3)
                with norm_col1:
                    st.metric("Test Used", test_name)
                with norm_col2:
                    st.metric("P-value", f"{p_diff:.4f}")
                with norm_col3:
                    if is_normal:
                        st.success("✅ Data is Normal")
                    else:
                        st.warning("⚠️ Data is Non-Normal")
                
                # Interpretation
                if is_normal:
                    st.info(f"**Interpretation:** Difference scores appear normally distributed (p = {p_diff:.3f} > {alpha}). **Using Paired t-test.**")
                    statistical_test = "Paired t-test"
                else:
                    st.warning(f"**Interpretation:** Difference scores are not normally distributed (p = {p_diff:.3f} ≤ {alpha}). **Using Wilcoxon signed-rank test** (non-parametric alternative).")
                    statistical_test = "Wilcoxon signed-rank test"
                
                # Visual normality check
                norm_viz_col1, norm_viz_col2 = st.columns(2)
                
                with norm_viz_col1:
                    # Histogram with normal curve
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(
                        x=difference,
                        name='Differences',
                        nbinsx=min(20, n//2),
                        histnorm='probability density'
                    ))
                    
                    # Add normal curve
                    x_range = np.linspace(difference.min(), difference.max(), 100)
                    normal_curve = stats.norm.pdf(x_range, difference.mean(), difference.std())
                    fig_hist.add_trace(go.Scatter(
                        x=x_range,
                        y=normal_curve,
                        name='Normal Distribution',
                        line=dict(color='red', width=2)
                    ))
                    
                    fig_hist.update_layout(
                        title="Distribution of Differences",
                        xaxis_title="Difference (Post - Pre)",
                        yaxis_title="Density",
                        height=400
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with norm_viz_col2:
                    # Q-Q plot
                    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, n))
                    sample_quantiles = np.sort(difference)
                    
                    fig_qq = go.Figure()
                    fig_qq.add_trace(go.Scatter(
                        x=theoretical_quantiles,
                        y=sample_quantiles,
                        mode='markers',
                        name='Sample Data',
                        marker=dict(size=8)
                    ))
                    
                    # Add reference line
                    line_min = min(theoretical_quantiles.min(), sample_quantiles.min())
                    line_max = max(theoretical_quantiles.max(), sample_quantiles.max())
                    fig_qq.add_trace(go.Scatter(
                        x=[line_min, line_max],
                        y=[line_min, line_max],
                        mode='lines',
                        name='Normal Reference',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig_qq.update_layout(
                        title="Q-Q Plot",
                        xaxis_title="Theoretical Quantiles",
                        yaxis_title="Sample Quantiles",
                        height=400
                    )
                    st.plotly_chart(fig_qq, use_container_width=True)
                
                # === STATISTICAL TEST ===
                st.markdown("---")
                st.header(f"📋 Results: {statistical_test}")
                
                # Perform appropriate test
                if is_normal:
                    # PAIRED T-TEST
                    t_stat, p_value = stats.ttest_rel(pre_data, post_data)
                    test_statistic = t_stat
                    test_stat_name = "t-statistic"
                    
                    # Effect size (Cohen's d for paired samples)
                    cohens_d = difference.mean() / difference.std()
                    effect_size_name = "Cohen's d"
                    effect_size = cohens_d
                    
                    # Confidence interval
                    se = difference.std() / np.sqrt(n)
                    ci_95 = stats.t.interval(0.95, n-1, loc=difference.mean(), scale=se)
                    
                else:
                    # WILCOXON SIGNED-RANK TEST
                    wilcoxon_result = stats.wilcoxon(pre_data, post_data, alternative='two-sided')
                    test_statistic = wilcoxon_result.statistic
                    p_value = wilcoxon_result.pvalue
                    test_stat_name = "W-statistic"
                    
                    # Effect size (r = Z / sqrt(N))
                    z_score = stats.norm.ppf(1 - p_value/2)  # approximate Z
                    r_effect = abs(z_score) / np.sqrt(n)
                    effect_size_name = "Effect size r"
                    effect_size = r_effect
                    
                    # Confidence interval (using Hodges-Lehmann estimator)
                    # This is the median of pairwise differences
                    ci_95 = None  # Wilcoxon CI is complex, showing median instead
                
                # Basic calculations
                pre_mean = pre_data.mean()
                post_mean = post_data.mean()
                pre_median = pre_data.median()
                post_median = post_data.median()
                pre_sd = pre_data.std()
                post_sd = post_data.std()
                mean_diff = difference.mean()
                median_diff = difference.median()
                sd_diff = difference.std()
                
                # Key metrics
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                with metric_col1:
                    st.metric("Sample Size (n)", n)
                with metric_col2:
                    if is_normal:
                        st.metric("Mean Difference", f"{mean_diff:.2f}", 
                                 delta=f"{((mean_diff/pre_mean)*100):.1f}%" if pre_mean != 0 else None)
                    else:
                        st.metric("Median Difference", f"{median_diff:.2f}")
                with metric_col3:
                    st.metric("P-value", f"{p_value:.4f}")
                with metric_col4:
                    if p_value < 0.001:
                        sig_text = "*** p<0.001"
                    elif p_value < 0.01:
                        sig_text = "** p<0.01"
                    elif p_value < alpha:
                        sig_text = f"* p<{alpha}"
                    else:
                        sig_text = "Not Significant"
                    st.metric("Significance", sig_text)
                
                # Detailed statistics table
                st.subheader("📊 Descriptive Statistics")
                if is_normal:
                    stats_df = pd.DataFrame({
                        'Measure': ['Pre', 'Post', 'Change'],
                        'Mean': [pre_mean, post_mean, mean_diff],
                        'SD': [pre_sd, post_sd, sd_diff],
                        'Median': [pre_median, post_median, median_diff],
                        'Min': [pre_data.min(), post_data.min(), difference.min()],
                        'Max': [pre_data.max(), post_data.max(), difference.max()]
                    })
                else:
                    stats_df = pd.DataFrame({
                        'Measure': ['Pre', 'Post', 'Change'],
                        'Median': [pre_median, post_median, median_diff],
                        'Mean': [pre_mean, post_mean, mean_diff],
                        'SD': [pre_sd, post_sd, sd_diff],
                        'Min': [pre_data.min(), post_data.min(), difference.min()],
                        'Max': [pre_data.max(), post_data.max(), difference.max()]
                    })
                
                st.dataframe(stats_df.style.format({
                    'Mean': '{:.2f}',
                    'Median': '{:.2f}',
                    'SD': '{:.2f}',
                    'Min': '{:.2f}',
                    'Max': '{:.2f}'
                }), use_container_width=True)
                
                # Test statistics
                st.subheader("🔬 Test Statistics")
                test_col1, test_col2 = st.columns(2)
                with test_col1:
                    st.metric(test_stat_name, f"{test_statistic:.3f}")
                    st.metric(effect_size_name, f"{effect_size:.3f}")
                with test_col2:
                    # Effect size interpretation
                    if is_normal:
                        if abs(effect_size) < 0.2:
                            effect_interpretation = "Negligible effect"
                        elif abs(effect_size) < 0.5:
                            effect_interpretation = "Small effect"
                        elif abs(effect_size) < 0.8:
                            effect_interpretation = "Medium effect"
                        else:
                            effect_interpretation = "Large effect"
                    else:
                        if abs(effect_size) < 0.1:
                            effect_interpretation = "Small effect"
                        elif abs(effect_size) < 0.3:
                            effect_interpretation = "Medium effect"
                        else:
                            effect_interpretation = "Large effect"
                    
                    st.info(f"**Interpretation:** {effect_interpretation}")
                    
                    if is_normal and ci_95:
                        st.metric("95% CI", f"[{ci_95[0]:.2f}, {ci_95[1]:.2f}]")
                
                # Visualizations
                st.markdown("---")
                st.header("📈 Visualizations")
                
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    # Bar chart comparing means/medians
                    if is_normal:
                        fig_bar = go.Figure(data=[
                            go.Bar(name='Pre', x=['Pre'], y=[pre_mean], 
                                  error_y=dict(type='data', array=[pre_sd])),
                            go.Bar(name='Post', x=['Post'], y=[post_mean], 
                                  error_y=dict(type='data', array=[post_sd]))
                        ])
                        fig_bar.update_layout(title="Mean Scores (Pre vs Post)", yaxis_title="Score")
                    else:
                        fig_bar = go.Figure(data=[
                            go.Bar(name='Pre', x=['Pre'], y=[pre_median]),
                            go.Bar(name='Post', x=['Post'], y=[post_median])
                        ])
                        fig_bar.update_layout(title="Median Scores (Pre vs Post)", yaxis_title="Score")
                    
                    fig_bar.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                with viz_col2:
                    # Box plot
                    fig_box = go.Figure()
                    fig_box.add_trace(go.Box(y=pre_data, name='Pre', marker_color='lightblue'))
                    fig_box.add_trace(go.Box(y=post_data, name='Post', marker_color='lightgreen'))
                    fig_box.update_layout(
                        title="Distribution Comparison",
                        yaxis_title="Score",
                        showlegend=True,
                        height=400
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
                
                # Individual changes
                st.subheader("👥 Individual Changes")
                change_data = pd.DataFrame({
                    'Participant': range(1, n+1),
                    'Pre': pre_data.values,
                    'Post': post_data.values,
                    'Change': difference.values
                })
                
                fig_individual = go.Figure()
                for i in range(n):
                    color = 'green' if difference.iloc[i] > 0 else 'red'
                    fig_individual.add_trace(go.Scatter(
                        x=['Pre', 'Post'],
                        y=[pre_data.iloc[i], post_data.iloc[i]],
                        mode='lines+markers',
                        name=f'P{i+1}',
                        line=dict(color=f'rgba(100,100,100,0.3)'),
                        showlegend=False,
                        hovertemplate=f'Participant {i+1}<br>Change: {difference.iloc[i]:.1f}<extra></extra>'
                    ))
                
                # Add mean/median line
                if is_normal:
                    fig_individual.add_trace(go.Scatter(
                        x=['Pre', 'Post'],
                        y=[pre_mean, post_mean],
                        mode='lines+markers',
                        name='Mean',
                        line=dict(color='red', width=3),
                        marker=dict(size=10)
                    ))
                else:
                    fig_individual.add_trace(go.Scatter(
                        x=['Pre', 'Post'],
                        y=[pre_median, post_median],
                        mode='lines+markers',
                        name='Median',
                        line=dict(color='red', width=3),
                        marker=dict(size=10)
                    ))
                
                fig_individual.update_layout(
                    title="Individual Trajectories",
                    yaxis_title="Score",
                    height=500
                )
                st.plotly_chart(fig_individual, use_container_width=True)
                
                # Download report
                st.markdown("---")
                st.header("💾 Download Report")
                
                # Create Excel report
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # Summary sheet
                    summary_data = {
                        'Statistic': [
                            'Sample Size', 
                            'Statistical Test Used',
                            'Normality Test',
                            'Normality P-value',
                            'Data Distribution',
                            'Pre Mean', 'Pre SD', 'Pre Median',
                            'Post Mean', 'Post SD', 'Post Median',
                            'Mean Difference', 'Median Difference', 'SD Difference', 
                            test_stat_name, 
                            'p-value', 
                            effect_size_name
                        ],
                        'Value': [
                            n, 
                            statistical_test,
                            test_name,
                            p_diff,
                            'Normal' if is_normal else 'Non-normal',
                            pre_mean, pre_sd, pre_median,
                            post_mean, post_sd, post_median,
                            mean_diff, median_diff, sd_diff, 
                            test_statistic, 
                            p_value, 
                            effect_size
                        ]
                    }
                    
                    if is_normal and ci_95:
                        summary_data['Statistic'].extend(['95% CI Lower', '95% CI Upper'])
                        summary_data['Value'].extend([ci_95[0], ci_95[1]])
                    
                    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                    
                    # Individual data
                    change_data.to_excel(writer, sheet_name='Individual_Data', index=False)
                    
                    # Raw data
                    df.to_excel(writer, sheet_name='Raw_Data', index=False)
                
                output.seek(0)
                
                st.download_button(
                    label="📥 Download Excel Report",
                    data=output,
                    file_name="prepost_analysis_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
                
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please check that your file format is correct and contains numeric data.")
        st.exception(e)  # Show detailed error for debugging

else:
    st.info("👈 Please upload a file to begin analysis")
    
    # Show example download
    st.markdown("### 📥 Download Example Template")
    example_template = pd.DataFrame({
        'Participant_ID': range(1, 21),
        'Pre_Score': np.random.randint(50, 80, 20),
        'Post_Score': np.random.randint(60, 95, 20)
    })
    
    template_buffer = BytesIO()
    example_template.to_excel(template_buffer, index=False)
    template_buffer.seek(0)
    
    st.download_button(
        label="Download Example Template",
        data=template_buffer,
        file_name="prepost_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <small>Pre-Post Comparison Tool | Automatic Test Selection | Made with Streamlit</small>
</div>
""", unsafe_allow_html=True)