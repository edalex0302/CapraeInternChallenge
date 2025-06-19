import numpy as np

def score_lead(lead_row, model, scaler):
    """
    Calculates lead score and assigns rarity tier.
    Returns the score, a deep dive and a tier with color.
    Generate highlitghts of company based of scores

    """

    # Extracting features
    features = [
        lead_row['platform_count'],
        lead_row['avg_activity_days'],
        lead_row['linkedin_exists'],
        lead_row['twitter_exists'],
        lead_row['instagram_exists'],
        lead_row['linkedin_activity_days'],
        lead_row['twitter_activity_days'],
        lead_row['instagram_activity_days']
    ]

    # model score (0–100)
    X = scaler.transform([features])
    ml_prob = model.predict_proba(X)[0][1]
    ml_score = ml_prob * 100

    # scoring manually based on result existing and active
    platform_points = (
        lead_row['linkedin_exists'] +
        lead_row['twitter_exists'] +
        lead_row['instagram_exists']
    ) * 5

    activity_days = np.array([
        lead_row['linkedin_activity_days'],
        lead_row['twitter_activity_days'],
        lead_row['instagram_activity_days']
    ])
    activity_score = np.clip(30 - activity_days, 0, 30).sum() / 3
    activity_points = (activity_score / 30) * 15

    # Final score resulttt
    final_score = ml_score * 0.7 + platform_points + activity_points
    final_score = round(min(final_score, 100), 1)

    # devide to rarity colour tier
    def get_rarity(score):
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

    rarity_label, rarity_color = get_rarity(final_score)

    # Return ze values
    breakdown = {
        "ml_score (70%)": round(ml_score * 0.7, 1),
        "platform_points (max 15)": platform_points,
        "activity_points (max 15)": round(activity_points, 1),
    }

    return final_score, breakdown, rarity_label, rarity_color

#for reason for company to be highligthed
def generate_reason(row):
    reasons = []

    # Social Media Activiteness Analysis 
    platforms = [
        ("has_twitter", "twitter_last_post_days_ago"),
        ("has_linkedin", "linkedin_last_post_days_ago"),
        ("has_tiktok", None),
        ("has_facebook", "facebook_last_post_days_ago"),
    ]

    total_platforms = 0
    active_recently = 0

    for has_key, days_key in platforms:
        if row.get(has_key):
            total_platforms += 1
            if days_key and row.get(days_key, 999) < 30:
                active_recently += 1
            elif days_key is None:
                # case platform like tik tok doesnt have recency 
                active_recently += 1

    if total_platforms >= 3 and active_recently >= 3:
        reasons.append("📢 Active on multiple social media platforms")
    elif active_recently >= 1:
        reasons.append("📣 Active on social media")

    # Other Traits to Highlight
    if row.get("website_exists"):
        reasons.append("🌐 Has a company website")

    if row.get("founder_linkedin_exists"):
        reasons.append("🧑‍💼 Founder has a LinkedIn profile")

    if row.get("employee_count", 0) > 50:
        reasons.append(f"👥 {row.get('employee_count')} employees")

    if row.get("estimated_revenue", 0) > 500000:
        reasons.append(f"💰 Estimated revenue ${row.get('estimated_revenue'):,}")

    if row.get("bbb_rating") and isinstance(row["bbb_rating"], (int, float)) and row["bbb_rating"] >= 85:
        reasons.append("🏅 Strong BBB rating")
    #else
    if not reasons:
        return "No standout traits detected."

    return "• " + "\n• ".join(reasons)

