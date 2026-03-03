from src.metrics import calculate_dv


def run_hcm_search(df, seed_id, v_limit):
    """Finds all asteroids within the velocity cutoff of the seed"""
    seed_row = df[df["id"] == seed_id].iloc[0]
    family_members = []

    for _, row in df.iterrows():
        if row["id"] == seed_id:
            continue

        # Calculate distance in velocity space
        dist = calculate_dv(seed_row, row)
        if dist < v_limit:
            family_members.append(row["id"])

    return family_members
