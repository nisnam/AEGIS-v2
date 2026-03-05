"""
Generate realistic synthetic arms trade data for AEGIS dashboard.
Aligned with SIPRI Arms Transfers Database patterns.
"""
import pandas as pd
import numpy as np

np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# COUNTRY PROFILES
# ─────────────────────────────────────────────────────────────

EXPORTER_PROFILES = {
    'United States':  {'region': 'North America',     'alliance': 'NATO',         'unsc': 'Yes', 'share': 0.22},
    'Russia':         {'region': 'Europe & Central Asia','alliance': 'Non-Aligned','unsc': 'Yes', 'share': 0.17},
    'France':         {'region': 'Europe & Central Asia','alliance': 'NATO',       'unsc': 'Yes', 'share': 0.12},
    'China':          {'region': 'East Asia & Pacific', 'alliance': 'Non-Aligned', 'unsc': 'Yes', 'share': 0.10},
    'Germany':        {'region': 'Europe & Central Asia','alliance': 'NATO',       'unsc': 'No',  'share': 0.07},
    'Italy':          {'region': 'Europe & Central Asia','alliance': 'NATO',       'unsc': 'No',  'share': 0.05},
    'United Kingdom': {'region': 'Europe & Central Asia','alliance': 'NATO',       'unsc': 'Yes', 'share': 0.05},
    'South Korea':    {'region': 'East Asia & Pacific', 'alliance': 'Non-NATO Ally','unsc': 'No', 'share': 0.05},
    'Israel':         {'region': 'Middle East & North Africa','alliance': 'Non-NATO Ally','unsc': 'No','share': 0.05},
    'Turkey':         {'region': 'Europe & Central Asia','alliance': 'NATO',       'unsc': 'No',  'share': 0.04},
    'Spain':          {'region': 'Europe & Central Asia','alliance': 'NATO',       'unsc': 'No',  'share': 0.03},
    'Sweden':         {'region': 'Europe & Central Asia','alliance': 'NATO',       'unsc': 'No',  'share': 0.05},
}

IMPORTER_PROFILES = {
    # Heavy importers
    'India':        {'region': 'South Asia',               'gdp': 2500,  'stability': 4.8, 'democracy': 6.6, 'conflict': 'Yes', 'dispute': 'Yes', 'resource': 'Low',    'mil_spend': 2.4, 'trend_weights': [0.30, 0.45, 0.25], 'embargo': 'No',  'weight': 110},
    'Saudi Arabia': {'region': 'Middle East & North Africa','gdp': 23000, 'stability': 4.2, 'democracy': 1.8, 'conflict': 'Yes', 'dispute': 'No',  'resource': 'High',   'mil_spend': 6.0, 'trend_weights': [0.40, 0.40, 0.20], 'embargo': 'No',  'weight': 110},
    # Major importers
    'Qatar':        {'region': 'Middle East & North Africa','gdp': 60000, 'stability': 7.0, 'democracy': 2.0, 'conflict': 'No',  'dispute': 'No',  'resource': 'High',   'mil_spend': 1.5, 'trend_weights': [0.35, 0.45, 0.20], 'embargo': 'No',  'weight': 80},
    'Egypt':        {'region': 'Middle East & North Africa','gdp': 3500,  'stability': 3.8, 'democracy': 2.8, 'conflict': 'Yes', 'dispute': 'No',  'resource': 'Low',    'mil_spend': 1.3, 'trend_weights': [0.25, 0.50, 0.25], 'embargo': 'No',  'weight': 80},
    'Australia':    {'region': 'East Asia & Pacific',      'gdp': 55000, 'stability': 8.2, 'democracy': 8.7, 'conflict': 'No',  'dispute': 'No',  'resource': 'Medium', 'mil_spend': 2.0, 'trend_weights': [0.30, 0.50, 0.20], 'embargo': 'No',  'weight': 80},
    'Pakistan':     {'region': 'South Asia',               'gdp': 1500,  'stability': 3.2, 'democracy': 4.1, 'conflict': 'Yes', 'dispute': 'Yes', 'resource': 'Low',    'mil_spend': 3.8, 'trend_weights': [0.35, 0.40, 0.25], 'embargo': 'No',  'weight': 80},
    # Significant importers
    'South Korea':  {'region': 'East Asia & Pacific',      'gdp': 32000, 'stability': 7.5, 'democracy': 8.0, 'conflict': 'No',  'dispute': 'Yes', 'resource': 'Low',    'mil_spend': 2.6, 'trend_weights': [0.20, 0.55, 0.25], 'embargo': 'No',  'weight': 55},
    'Japan':        {'region': 'East Asia & Pacific',      'gdp': 40000, 'stability': 8.5, 'democracy': 8.1, 'conflict': 'No',  'dispute': 'Yes', 'resource': 'Low',    'mil_spend': 1.2, 'trend_weights': [0.30, 0.50, 0.20], 'embargo': 'No',  'weight': 55},
    'UAE':          {'region': 'Middle East & North Africa','gdp': 43000, 'stability': 6.8, 'democracy': 2.5, 'conflict': 'Yes', 'dispute': 'No',  'resource': 'High',   'mil_spend': 5.6, 'trend_weights': [0.35, 0.40, 0.25], 'embargo': 'No',  'weight': 55},
    'Indonesia':    {'region': 'East Asia & Pacific',      'gdp': 4200,  'stability': 5.2, 'democracy': 6.3, 'conflict': 'No',  'dispute': 'Yes', 'resource': 'Medium', 'mil_spend': 0.8, 'trend_weights': [0.25, 0.50, 0.25], 'embargo': 'No',  'weight': 55},
    'Algeria':      {'region': 'Middle East & North Africa','gdp': 3700,  'stability': 3.5, 'democracy': 3.2, 'conflict': 'No',  'dispute': 'Yes', 'resource': 'High',   'mil_spend': 5.3, 'trend_weights': [0.20, 0.50, 0.30], 'embargo': 'No',  'weight': 55},
    # Moderate importers
    'Bangladesh':   {'region': 'South Asia',               'gdp': 2200,  'stability': 4.0, 'democracy': 4.8, 'conflict': 'No',  'dispute': 'No',  'resource': 'Low',    'mil_spend': 1.2, 'trend_weights': [0.20, 0.55, 0.25], 'embargo': 'No',  'weight': 35},
    'Vietnam':      {'region': 'East Asia & Pacific',      'gdp': 3600,  'stability': 5.5, 'democracy': 2.9, 'conflict': 'No',  'dispute': 'Yes', 'resource': 'Low',    'mil_spend': 2.3, 'trend_weights': [0.25, 0.50, 0.25], 'embargo': 'No',  'weight': 35},
    'Thailand':     {'region': 'East Asia & Pacific',      'gdp': 7200,  'stability': 5.0, 'democracy': 4.6, 'conflict': 'No',  'dispute': 'No',  'resource': 'Low',    'mil_spend': 1.4, 'trend_weights': [0.15, 0.55, 0.30], 'embargo': 'No',  'weight': 35},
    'Singapore':    {'region': 'East Asia & Pacific',      'gdp': 65000, 'stability': 9.0, 'democracy': 6.0, 'conflict': 'No',  'dispute': 'No',  'resource': 'Low',    'mil_spend': 3.2, 'trend_weights': [0.20, 0.60, 0.20], 'embargo': 'No',  'weight': 35},
    'Taiwan':       {'region': 'East Asia & Pacific',      'gdp': 32000, 'stability': 7.2, 'democracy': 8.9, 'conflict': 'No',  'dispute': 'Yes', 'resource': 'Low',    'mil_spend': 2.3, 'trend_weights': [0.40, 0.45, 0.15], 'embargo': 'No',  'weight': 35},
    'Poland':       {'region': 'Europe & Central Asia',    'gdp': 17000, 'stability': 7.0, 'democracy': 7.0, 'conflict': 'No',  'dispute': 'No',  'resource': 'Low',    'mil_spend': 3.9, 'trend_weights': [0.45, 0.40, 0.15], 'embargo': 'No',  'weight': 35},
    'Greece':       {'region': 'Europe & Central Asia',    'gdp': 20000, 'stability': 6.2, 'democracy': 7.4, 'conflict': 'No',  'dispute': 'Yes', 'resource': 'Low',    'mil_spend': 3.7, 'trend_weights': [0.20, 0.55, 0.25], 'embargo': 'No',  'weight': 35},
    'Norway':       {'region': 'Europe & Central Asia',    'gdp': 78000, 'stability': 8.8, 'democracy': 9.8, 'conflict': 'No',  'dispute': 'No',  'resource': 'High',   'mil_spend': 1.7, 'trend_weights': [0.25, 0.55, 0.20], 'embargo': 'No',  'weight': 20},
    'Colombia':     {'region': 'Latin America & Caribbean','gdp': 6500,  'stability': 3.9, 'democracy': 6.0, 'conflict': 'Yes', 'dispute': 'No',  'resource': 'Medium', 'mil_spend': 3.2, 'trend_weights': [0.15, 0.55, 0.30], 'embargo': 'No',  'weight': 25},
    'Morocco':      {'region': 'Middle East & North Africa','gdp': 3500,  'stability': 4.5, 'democracy': 3.8, 'conflict': 'No',  'dispute': 'Yes', 'resource': 'Medium', 'mil_spend': 3.1, 'trend_weights': [0.30, 0.50, 0.20], 'embargo': 'No',  'weight': 25},
    # Minor importers
    'Ukraine':      {'region': 'Europe & Central Asia',    'gdp': 3800,  'stability': 2.5, 'democracy': 5.4, 'conflict': 'Yes', 'dispute': 'Yes', 'resource': 'Low',    'mil_spend': 4.5, 'trend_weights': [0.55, 0.35, 0.10], 'embargo': 'No',  'weight': 25},
    'Brazil':       {'region': 'Latin America & Caribbean','gdp': 8700,  'stability': 5.5, 'democracy': 6.9, 'conflict': 'No',  'dispute': 'No',  'resource': 'Medium', 'mil_spend': 1.3, 'trend_weights': [0.10, 0.55, 0.35], 'embargo': 'No',  'weight': 20},
    'Philippines':  {'region': 'East Asia & Pacific',      'gdp': 3500,  'stability': 4.2, 'democracy': 6.1, 'conflict': 'Yes', 'dispute': 'Yes', 'resource': 'Low',    'mil_spend': 1.1, 'trend_weights': [0.30, 0.50, 0.20], 'embargo': 'No',  'weight': 20},
    'Mexico':       {'region': 'Latin America & Caribbean','gdp': 10000, 'stability': 3.8, 'democracy': 5.6, 'conflict': 'No',  'dispute': 'No',  'resource': 'Medium', 'mil_spend': 0.5, 'trend_weights': [0.15, 0.55, 0.30], 'embargo': 'No',  'weight': 15},
    'Nigeria':      {'region': 'Sub-Saharan Africa',       'gdp': 2100,  'stability': 2.8, 'democracy': 4.0, 'conflict': 'Yes', 'dispute': 'Yes', 'resource': 'High',   'mil_spend': 0.6, 'trend_weights': [0.25, 0.50, 0.25], 'embargo': 'No',  'weight': 18},
    'Ethiopia':     {'region': 'Sub-Saharan Africa',       'gdp': 900,   'stability': 2.2, 'democracy': 3.0, 'conflict': 'Yes', 'dispute': 'Yes', 'resource': 'Low',    'mil_spend': 0.7, 'trend_weights': [0.35, 0.45, 0.20], 'embargo': 'Yes', 'weight': 18},
    'Kenya':        {'region': 'Sub-Saharan Africa',       'gdp': 1800,  'stability': 3.5, 'democracy': 4.5, 'conflict': 'Yes', 'dispute': 'No',  'resource': 'Low',    'mil_spend': 1.1, 'trend_weights': [0.20, 0.55, 0.25], 'embargo': 'No',  'weight': 15},
    'Angola':       {'region': 'Sub-Saharan Africa',       'gdp': 2000,  'stability': 3.0, 'democracy': 2.6, 'conflict': 'No',  'dispute': 'No',  'resource': 'High',   'mil_spend': 1.6, 'trend_weights': [0.15, 0.50, 0.35], 'embargo': 'No',  'weight': 15},
    'Peru':         {'region': 'Latin America & Caribbean','gdp': 6600,  'stability': 4.5, 'democracy': 6.2, 'conflict': 'No',  'dispute': 'No',  'resource': 'Medium', 'mil_spend': 1.1, 'trend_weights': [0.10, 0.55, 0.35], 'embargo': 'No',  'weight': 12},
    'Kazakhstan':   {'region': 'Europe & Central Asia',    'gdp': 9800,  'stability': 4.8, 'democracy': 2.9, 'conflict': 'No',  'dispute': 'No',  'resource': 'High',   'mil_spend': 1.1, 'trend_weights': [0.15, 0.55, 0.30], 'embargo': 'No',  'weight': 15},
    'Chile':        {'region': 'Latin America & Caribbean','gdp': 15000, 'stability': 6.5, 'democracy': 8.0, 'conflict': 'No',  'dispute': 'No',  'resource': 'Medium', 'mil_spend': 1.8, 'trend_weights': [0.10, 0.55, 0.35], 'embargo': 'No',  'weight': 12},
    'Romania':      {'region': 'Europe & Central Asia',    'gdp': 14000, 'stability': 5.8, 'democracy': 6.4, 'conflict': 'No',  'dispute': 'No',  'resource': 'Low',    'mil_spend': 2.4, 'trend_weights': [0.35, 0.50, 0.15], 'embargo': 'No',  'weight': 12},
    'Iraq':         {'region': 'Middle East & North Africa','gdp': 4700,  'stability': 2.0, 'democracy': 3.5, 'conflict': 'Yes', 'dispute': 'Yes', 'resource': 'High',   'mil_spend': 3.5, 'trend_weights': [0.30, 0.45, 0.25], 'embargo': 'No',  'weight': 25},
}

# ─────────────────────────────────────────────────────────────
# BILATERAL FLOW RULES (exporter -> {importer: relative_weight})
# ─────────────────────────────────────────────────────────────

FLOWS = {
    'United States': {
        'Saudi Arabia': 40, 'Australia': 35, 'Japan': 30, 'South Korea': 30, 'Taiwan': 25,
        'Egypt': 25, 'UAE': 20, 'Poland': 20, 'India': 15, 'Qatar': 15,
        'Israel': 0, 'Singapore': 10, 'Greece': 10, 'Norway': 8, 'Philippines': 8,
        'Colombia': 8, 'Morocco': 6, 'Indonesia': 6, 'Thailand': 5, 'Brazil': 4,
        'Iraq': 15, 'Ukraine': 10, 'Romania': 5, 'Chile': 3, 'Mexico': 5,
        'Kenya': 3, 'Nigeria': 3, 'Bangladesh': 2,
    },
    'Russia': {
        'India': 70, 'China': 0, 'Algeria': 30, 'Egypt': 20, 'Vietnam': 20,
        'Bangladesh': 15, 'Kazakhstan': 12, 'Indonesia': 10, 'Pakistan': 10,
        'Ethiopia': 8, 'Angola': 8, 'Nigeria': 6, 'Iraq': 10, 'Myanmar': 0,
        'Thailand': 5, 'Peru': 4, 'Mexico': 4,
    },
    'France': {
        'India': 25, 'Egypt': 25, 'Qatar': 25, 'Saudi Arabia': 20, 'UAE': 15,
        'Australia': 12, 'Greece': 10, 'Singapore': 10, 'Brazil': 8,
        'Indonesia': 8, 'Morocco': 8, 'Colombia': 6, 'Poland': 4, 'Chile': 4,
        'Malaysia': 0, 'Peru': 3,
    },
    'China': {
        'Pakistan': 40, 'Bangladesh': 25, 'Thailand': 15, 'Algeria': 12,
        'Nigeria': 10, 'Kazakhstan': 10, 'Ethiopia': 8, 'Kenya': 8,
        'Angola': 6, 'Indonesia': 6, 'Vietnam': 5, 'Iraq': 5,
    },
    'Germany': {
        'South Korea': 15, 'Greece': 12, 'Australia': 10, 'Egypt': 10,
        'Singapore': 8, 'Indonesia': 8, 'Norway': 6, 'Chile': 6,
        'Algeria': 5, 'Poland': 5, 'India': 5, 'Brazil': 3,
        'Qatar': 4, 'Romania': 3, 'Colombia': 3,
    },
    'Italy': {
        'Egypt': 12, 'Qatar': 10, 'UAE': 8, 'India': 6, 'Pakistan': 5,
        'Poland': 5, 'Singapore': 4, 'Brazil': 4, 'Nigeria': 3,
        'Thailand': 3, 'Philippines': 3, 'Morocco': 3, 'Colombia': 3,
    },
    'United Kingdom': {
        'Saudi Arabia': 20, 'Qatar': 10, 'UAE': 8, 'India': 6, 'Australia': 6,
        'Japan': 4, 'Poland': 4, 'South Korea': 3, 'Singapore': 3,
        'Indonesia': 3, 'Kenya': 2, 'Nigeria': 2, 'Chile': 2, 'Romania': 2,
    },
    'South Korea': {
        'Indonesia': 12, 'Philippines': 10, 'Thailand': 8, 'India': 8,
        'Poland': 8, 'Australia': 6, 'Norway': 5, 'Egypt': 4,
        'Colombia': 4, 'Peru': 3, 'Romania': 3, 'Chile': 3, 'Iraq': 3,
    },
    'Israel': {
        'India': 20, 'Singapore': 8, 'South Korea': 6, 'Australia': 5,
        'Vietnam': 5, 'Philippines': 5, 'Brazil': 4, 'Colombia': 4,
        'Morocco': 4, 'Greece': 3, 'UAE': 3, 'Thailand': 3, 'Chile': 3,
        'Kazakhstan': 2, 'Romania': 2,
    },
    'Turkey': {
        'Pakistan': 10, 'Qatar': 6, 'UAE': 5, 'Bangladesh': 5,
        'Philippines': 4, 'Ukraine': 8, 'Poland': 4, 'Indonesia': 4,
        'Nigeria': 3, 'Morocco': 3, 'Kazakhstan': 3, 'Romania': 3,
    },
    'Spain': {
        'Australia': 8, 'Saudi Arabia': 5, 'Norway': 4, 'Brazil': 4,
        'Egypt': 4, 'Colombia': 3, 'Chile': 3, 'India': 3, 'Thailand': 3,
        'Singapore': 3, 'Peru': 2,
    },
    'Sweden': {
        'India': 10, 'Australia': 8, 'Brazil': 6, 'South Korea': 6,
        'Thailand': 5, 'Singapore': 5, 'Poland': 5, 'Norway': 4,
        'UAE': 4, 'Indonesia': 4, 'Greece': 4, 'Chile': 3,
        'Colombia': 3, 'Romania': 3,
    },
}

# ─────────────────────────────────────────────────────────────
# WEAPON DEFINITIONS
# ─────────────────────────────────────────────────────────────

WEAPON_CATEGORIES = {
    'Combat Aircraft': {
        'subtypes': [
            ('Fighter Jet', 'Offensive', 0.40),
            ('Multi-Role Aircraft', 'Offensive', 0.25),
            ('Trainer/Light Combat', 'Defensive', 0.20),
            ('UAV/Drone', 'Offensive', 0.15),
        ],
        'weight': 0.15,
        'value_range': (40, 900),
    },
    'Armoured Vehicles': {
        'subtypes': [
            ('Main Battle Tank', 'Offensive', 0.30),
            ('IFV', 'Offensive', 0.25),
            ('APC', 'Defensive', 0.25),
            ('MRAP', 'Defensive', 0.20),
        ],
        'weight': 0.18,
        'value_range': (5, 250),
    },
    'Missiles & Air Defence': {
        'subtypes': [
            ('SAM System', 'Defensive', 0.30),
            ('MANPADS', 'Defensive', 0.20),
            ('Cruise Missile', 'Offensive', 0.25),
            ('Anti-Ship Missile', 'Offensive', 0.25),
        ],
        'weight': 0.16,
        'value_range': (10, 600),
    },
    'Naval Vessels': {
        'subtypes': [
            ('Frigate', 'Offensive', 0.30),
            ('Submarine', 'Offensive', 0.25),
            ('Corvette', 'Offensive', 0.25),
            ('Patrol Boat', 'Defensive', 0.20),
        ],
        'weight': 0.13,
        'value_range': (20, 800),
    },
    'Artillery & Ammunition': {
        'subtypes': [
            ('MLRS', 'Offensive', 0.25),
            ('Self-Propelled Howitzer', 'Offensive', 0.25),
            ('Mortar System', 'Offensive', 0.25),
            ('Guided Munitions', 'Offensive', 0.25),
        ],
        'weight': 0.12,
        'value_range': (3, 200),
    },
    'Helicopters': {
        'subtypes': [
            ('Attack Helicopter', 'Offensive', 0.30),
            ('Utility Helicopter', 'Defensive', 0.30),
            ('Transport Helicopter', 'Defensive', 0.20),
            ('Naval Helicopter', 'Defensive', 0.20),
        ],
        'weight': 0.10,
        'value_range': (15, 400),
    },
    'Sensors & Electronics': {
        'subtypes': [
            ('Targeting Pod', 'Offensive', 0.25),
            ('C4ISR Package', 'Defensive', 0.25),
            ('EW Suite', 'Defensive', 0.25),
            ('Radar System', 'Defensive', 0.25),
        ],
        'weight': 0.09,
        'value_range': (5, 150),
    },
    'Small Arms & Light Weapons': {
        'subtypes': [
            ('ATGM', 'Offensive', 0.35),
            ('Sniper System', 'Offensive', 0.25),
            ('Assault Rifle Batch', 'Offensive', 0.25),
            ('Grenade Launcher Batch', 'Offensive', 0.15),
        ],
        'weight': 0.07,
        'value_range': (1, 40),
    },
}

DEAL_FRAMEWORKS = ['Foreign Military Sales', 'Bilateral Defence Pact', 'Government-to-Government',
                   'Direct Commercial Sale', 'Multilateral Programme', 'Offset Agreement']

# ─────────────────────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────────────────────

N = 1500
records = []

# Build exporter allocation
exporter_counts = {}
remaining = N
for exp, prof in list(EXPORTER_PROFILES.items())[:-1]:
    cnt = round(N * prof['share'])
    exporter_counts[exp] = cnt
    remaining -= cnt
last_exp = list(EXPORTER_PROFILES.keys())[-1]
exporter_counts[last_exp] = remaining

for exporter, n_deals in exporter_counts.items():
    exp_prof = EXPORTER_PROFILES[exporter]
    flow_map = FLOWS.get(exporter, {})

    # Filter to valid importers only
    valid_importers = {k: v for k, v in flow_map.items() if k in IMPORTER_PROFILES and v > 0}
    if not valid_importers:
        continue

    imp_names = list(valid_importers.keys())
    imp_weights = np.array(list(valid_importers.values()), dtype=float)
    imp_weights /= imp_weights.sum()

    # Assign importers
    importers = np.random.choice(imp_names, size=n_deals, p=imp_weights)

    for importer in importers:
        imp_prof = IMPORTER_PROFILES[importer]

        # Year (2005-2024)
        year = np.random.randint(2005, 2025)

        # Weapon category
        cat_names = list(WEAPON_CATEGORIES.keys())
        cat_weights = [WEAPON_CATEGORIES[c]['weight'] for c in cat_names]
        cat = np.random.choice(cat_names, p=cat_weights)
        cat_info = WEAPON_CATEGORIES[cat]

        # Weapon subtype
        sub_names = [s[0] for s in cat_info['subtypes']]
        sub_classes = [s[1] for s in cat_info['subtypes']]
        sub_weights = [s[2] for s in cat_info['subtypes']]
        sub_idx = np.random.choice(len(sub_names), p=sub_weights)
        subtype = sub_names[sub_idx]
        weapon_class = sub_classes[sub_idx]

        # Deal value (lognormal, clipped to category range)
        vmin, vmax = cat_info['value_range']
        raw_val = np.random.lognormal(mean=np.log((vmin + vmax) / 3), sigma=0.7)
        deal_value = round(np.clip(raw_val, vmin, vmax * 1.5), 1)

        # Quantity
        if cat in ['Small Arms & Light Weapons']:
            quantity = np.random.randint(50, 500)
        elif cat in ['Artillery & Ammunition', 'Missiles & Air Defence']:
            quantity = np.random.randint(2, 80)
        elif cat in ['Armoured Vehicles']:
            quantity = np.random.randint(5, 120)
        elif cat in ['Sensors & Electronics']:
            quantity = np.random.randint(1, 30)
        else:
            quantity = np.random.randint(1, 25)

        # Deal framework
        framework = np.random.choice(DEAL_FRAMEWORKS, p=[0.25, 0.20, 0.20, 0.15, 0.10, 0.10])

        # Technology transfer (more likely for big deals)
        tech_transfer = 'Yes' if (deal_value > 100 and np.random.random() < 0.4) else ('Yes' if np.random.random() < 0.15 else 'No')

        # Delivery timeline
        delivery = np.random.randint(6, 48)

        # Importer attributes (with jitter)
        gdp = round(imp_prof['gdp'] * np.random.uniform(0.92, 1.08))
        stability = round(np.clip(imp_prof['stability'] + np.random.uniform(-0.3, 0.3), 0.5, 10.0), 1)
        democracy = round(np.clip(imp_prof['democracy'] + np.random.uniform(-0.2, 0.2), 1.0, 10.0), 1)
        mil_spend = round(np.clip(imp_prof['mil_spend'] + np.random.uniform(-0.3, 0.3), 0.3, 8.0), 1)
        conflict = imp_prof['conflict']
        dispute = imp_prof['dispute']
        resource = imp_prof['resource']
        embargo = imp_prof['embargo']

        # Arms import trend
        trend = np.random.choice(['Accelerating', 'Stable', 'Declining'], p=imp_prof['trend_weights'])

        # ── Escalation Risk (deterministic logic) ──
        risk_score = 0
        risk_score += max(0, 5.0 - stability) * 2.5
        risk_score += max(0, 5.0 - democracy) * 1.0
        if conflict == 'Yes': risk_score += 6
        if dispute == 'Yes': risk_score += 4
        if weapon_class == 'Offensive': risk_score += 2
        if trend == 'Accelerating': risk_score += 3
        elif trend == 'Declining': risk_score -= 2
        if mil_spend > 4.0: risk_score += 3
        if embargo == 'Yes': risk_score += 8
        # Add some noise
        risk_score += np.random.normal(0, 2.5)

        if risk_score >= 12:
            esc_risk = 'High'
        elif risk_score >= 5:
            esc_risk = 'Medium'
        else:
            esc_risk = 'Low'

        records.append({
            'Year': year,
            'Exporter': exporter,
            'Exporter_Region': exp_prof['region'],
            'Exporter_Alliance': exp_prof['alliance'],
            'UNSC_Permanent_Member': exp_prof['unsc'],
            'Importer': importer,
            'Importer_Region': imp_prof['region'],
            'Weapon_Category': cat,
            'Weapon_Subtype': subtype,
            'Weapon_Class': weapon_class,
            'Deal_Value_USD_M': deal_value,
            'Quantity': quantity,
            'Deal_Framework': framework,
            'Technology_Transfer': tech_transfer,
            'Delivery_Timeline_Months': delivery,
            'Importer_GDP_Per_Capita': gdp,
            'Importer_Political_Stability': stability,
            'Importer_Democracy_Index': democracy,
            'Importer_Conflict_Proximity': conflict,
            'Active_Territorial_Dispute': dispute,
            'Natural_Resource_Dependence': resource,
            'Importer_Military_Spend_Pct_GDP': mil_spend,
            'Arms_Import_Trend': trend,
            'UN_Embargo': embargo,
            'Escalation_Risk': esc_risk,
        })

df = pd.DataFrame(records)

# Verify distributions
print(f"Total records: {len(df)}")
print(f"\nEscalation Risk distribution:")
print(df['Escalation_Risk'].value_counts(normalize=True).round(3))
print(f"\nTop 10 importers by value:")
print(df.groupby('Importer')['Deal_Value_USD_M'].sum().sort_values(ascending=False).head(10).round(0))
print(f"\nTop 5 exporters by deals:")
print(df['Exporter'].value_counts().head(5))
print(f"\nExporter Regions:")
print(df['Exporter_Region'].value_counts())
print(f"\nImporter Regions:")
print(df['Importer_Region'].value_counts())

df.to_csv("arms_trade.csv", index=False)
print(f"\nSaved to arms_trade.csv ({len(df)} rows x {len(df.columns)} cols)")
