import streamlit as st
import pandas as pd
import joblib
import numpy as np

# trained model
model = joblib.load("../model/Capraelead_score_model.pkl")

#wide for better reading view
st.set_page_config(page_title="Lead Quality Scorer", layout="wide")
st.title("🎯 Lead Quality Scoring Tool")

st.markdown("""
This tool assess the quality of leads based company Presence on social media platform .
""")

#  determine tier and color based on score
def get_score_color(score):
    if score >= 95:
        return "Best", "#ffa500"  # Gold
    elif score >= 90:
        return "Great", "#a020f0"       # Purple
    elif score >= 85:
        return "Good", "#1e90ff"       # Blue
    elif score >= 80:
        return "Fine", "#3cb371"   # Green
    elif score >= 75:
        return "Okay", "#a9a9a9"     # Gray
    else:
        return "Less than ideal", "#ff4c4c"   # Red

# csv upload
uploaded_file = st.file_uploader("Upload CSV==>", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    # using ml model
    try:
        # features in the model 
        features = [
            'has_twitter',
            'twitter_last_post_days_ago',
            'has_linkedin',
            'linkedin_last_post_days_ago',
            'employee_count',
            'estimated_revenue',
            'bbb_rating',
            'founder_linkedin_exists',
            'website_exists'
        ]

    # unerror missing columns
        missing = [f for f in features if f not in df.columns]
        if missing:
            st.error(f"Error: Uploaded CSV is missing required features: {missing}")
            st.stop()

        # define characters stopping error for bbb scoring
        rating_map = {
            'A+': 100, 'A': 95, 'A-': 90,
            'B+': 85, 'B': 80, 'B-': 75,
            'C+': 70, 'C': 65, 'C-': 60,
            'D+': 55, 'D': 50, 'D-': 45,
            'F': 40
        }
        df['bbb_rating'] = df['bbb_rating'].map(rating_map)

        # Select required features
        X = df[features]

        # Predict model
        scores = model.predict(X)

        df['lead_score'] = np.round(scores, 2)

        # Tier colour classification
        tiers, colors = [], []
        for s in df['lead_score']:
            tier, color = get_score_color(s)
            tiers.append(tier)
            colors.append(color)
        df['tier'] = tiers
        df['color'] = colors

        #import highlights from scoring
        from scoring import generate_reason
        reasons = []
        for _, row in df.iterrows():
            reasons.append(generate_reason(row))
        df["reason"] = reasons


        # the results
        st.subheader("Scored Leads Quality")
        for i, row in df.iterrows():
            reason = generate_reason(row)
            st.markdown(f"""
                <div style='padding: 10px; margin-bottom: 10px; border-radius: 10px; background-color: {row['color']}22;'>
                    <strong>{row.get('company_name', 'Company')}</strong> - Score: <strong>{row['lead_score']}</strong> 
                    <span style='float: right; background-color: {row['color']}; color: white; padding: 5px 10px; border-radius: 5px;'>{row['tier']}</span>
                    <br><small> Highlights <i>{reason}</i></small>
                </div>
            """, unsafe_allow_html=True)

        st.success("Done!")

        with st.expander("Download Scored Results"):
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name="scored_leads.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f" Error scoring leads: {e}")
        st.stop()

else:
    st.info(" Upload a CSV to start diving!.")

st.markdown("---")
st.caption("Built for Caprae Capital Intern Challenge")
