"""
ml/data_ingestion.py
--------------------
Fetches real french commune data from the INSEE public API,
then engineers health-related features based on DREES researhc

Data sources :
 - geo.api.gouv.fr -> commune names, codes, departments, population
 - DREES 2024 report -> GP density thresholds and desert definitions

 run this file directly to regenerate the dataset:
  python ml/data_ingestion.py
"""

import pandas as pd
import numpy as np
import requests
import json
from pathlib import Path
from app.config import settings

# ----- constants from DREES official research -------------
# National average GP density per 100,000 inhabitants (DREES 2024)
NATIONAL_AVG_GP_DENSITY = 147.0

# official threshold: 30% below national average = medical desert
DESERT_THRESHOLD = 0.70  # means gp density / national avg , 70%

# Departments classified as extremely rural by INSEE zoning
RURAL_DEPARTMENTS = [
    "23", "19", "15", "48", "43", "63",  # Massif Central
    "23", "36", "18", "58", "70", "52",  # Centre + haute-Saone
]

# Retirement Coast departments - high elderly, often high GP density
COASTAL_RETIREMENT_DEPARTMENTS = [
    "83", "06", "34", "30", "84", "13",   # Mediterranean coast
    "29", "56", "22", "35",               # Brittany
]

# Wealthy urban departments — generally well-served
WEALTHY_DEPARTMENTS = [
    "75", "92", "78", "77", "91",         # Paris + inner suburbs
    "94", "95", "93", "69", "67",         # Lyon, Strasbourg
]


def fetch_communes_from_insee() -> list[dict] | None:
    """
    Calls the insee GeoAPI to get all french communes.
    Returns a list of dicts or None if the API is unreachable

    The API is free, requires no authentication, and returns
    real data updated regularly by the French Government
    """

    print("Calling INSEE API.........")

    url = (
        "https://geo.api.gouv.fr/communes"
        "?fields=nom,code,codeDepartement,codeRegion,population"
        "&format=json"
    )

    try:
        # timeout -> 30 dont hangout forever if API is slow
        response = requests.get(url, timeout=30)

        if response.status_code == 200:
            data = response.json()
            print(f"data fetched of length {len(data):,} communes from INSEE")
            return data
        else:
            print(f"INSEE API returned status : {response.status_code}")
            return None
    except requests.exceptions.Timeout:
        print("Request time out after 30 seconds")
        return None

    except requests.exceptions.ConnectionError:
        print("Connection Error - No Internet Connection ")
        return None

    except Exception as e:
        print(f"Unexpected Error {e} ")
        return None


def build_synthetic_communes() -> list[dict]:
    """
      Fall back when INSEE API isnt reachable

      Generates a statistically sound sample of French Communes
      based on INSEE`s published distribution data
          - 45% of communes have population < 500 (tiny rural)
          - 30% have population 500-2000 (small rural)
          - 15% have population 2000-10000 (peri-urban)
          - 7%  have population 10000-50000 (small city)
          - 3%  have population > 50000 (city)

      Covers all major department types to ensure model
      sees the full spectrum of French healthcare situations.
    """

    print("synthetic communes based on INSEE distribution")

    np.random.seed(42)  # reproducable results
    # Representative departments covering all region types
    departments = [
        # (dept_code, region_code, type)
        ("75", "11"), ("92", "11"), ("93", "11"), ("94", "11"),  # Paris
        ("69", "84"), ("13", "93"), ("31", "76"), ("06", "93"),  # Major cities
        ("67", "44"), ("59", "32"), ("33", "75"), ("44", "52"),  # Regional Capitals
        ("23", "75"), ("19", "75"), ("15", "84"), ("48", "76"),  # Rural deserts
        ("03", "84"), ("36", "24"), ("58", "27"), ("43", "84"),  # Rural France
        ("83", "93"), ("34", "76"), ("30", "76"), ("84", "93"),  # South
        ("02", "32"), ("08", "44"), ("55", "44"), ("52", "44"),  # Northeast
        ("29", "53"), ("35", "53"), ("56", "53"), ("22", "53"),  # Brittany
    ]

     # Population size categories matching INSEE distribution
    pop_distribution = {
        "tiny":   (50,    500,   0.45),
        "small":  (500,   2000,  0.30),
        "medium": (2000,  10000, 0.15),
        "large":  (10000, 50000, 0.07),
        "city":   (50000, 500000, 0.03),
    }
    
    categories = list(pop_distribution.keys())
    weights = [pop_distribution[c][2] for c in categories]

    communes = []
    commune_id = 1

    for dept_code, region_code in departments:
        # more communes in urban area, few in rural areas
        n = np.random.randint(60, 120)

        for i in range(n):
            category = np.random.choice(categories, p=weights)
            low, high, _ = pop_distribution[category]
            population = int(np.random.randint(low, high))

            communes.append({
                "nom": f"Commune-{dept_code} - {i+1}",
                "code": f"{dept_code}{str(commune_id).zfill(4)}",
                "codeDepartment": dept_code,
                "codeRegion": region_code,
                "population": population
            })
            commune_id += 1

    print(f"✅ Generated {len(communes):,} synthetic communes")
    return communes

def classify_urban_score(population:int) -> int:
    """
        Converts population into urban classification score

        based on INSEE standard urban/rural typology:
            0 = rural          (< 2,000 inhabitants)
            1 = peri-urban     (2,000 __ 9,999)
            2 = small city     (10,000 __ 49,999)
            3 = urban          (50,000+)
    """

    if population < 2000:
        return 0
    elif population < 10000:
        return 1
    elif population < 50000:
        return 2
    else:
        return 3  
    
def engineer_features(communes: list[dict]) -> pd.DataFrame:
    """
        Transforms raw commune data into ML-Ready Features

        Feature Engineering = creating meaningful inputs for the model
        from raw data. THis is where domain knowledge matters the most
        Each Feature is justified by French Healthcare Research
    """
    print("Engineering Features")

    np.random.seed(42)
    records=[]

    for commune in communes:
        dept = commune.get("codeDepartment", "75")
        population = commune.get("population") or 0

        # skip communes with no meaningful population
        if population < 10:
            continue

        # Feature 1 : Population log
        # raw population is skewed ( Paris = 2M, tiny village = 50)
        # Log transforms compresses the scale -> better for ML
        # log(50) ≈ 3.9    log(2,000,000) ≈ 14.5
        # Without log, Paris would dominate the model completely
        population_log = np.log1p(population)

        # Feature 2 : Urban Score
        # 0-3 Scale, directly inked to healthcare access
        # rural areas have fewer doctors by definition
        urban_score = classify_urban_score(population)

        # Feature 3 : elderly_ratio
        # elderly population = higher healthcare demand
        # rural departments have much higher elderly ratios
        # source : INSEE regional age distributions 2023
        if dept in RURAL_DEPARTMENTS:
            elderly_ratio = np.random.uniform(0.28, 0.42)
        elif dept in COASTAL_RETIREMENT_DEPARTMENTS:
            elderly_ratio = np.random.uniform(0.24, 0.35)
        elif urban_score >=2 :
            elderly_ratio = np.random.uniform(0.13, 0.21)         
        else: 
            elderly_ratio = np.random.uniform(0.18, 0.30)

        # Feature 4 ----- Gp density per 100k
        # the most direct measure of healthcare access
        # deprived from DREES 2024 departmental level density data

        if dept in RURAL_DEPARTMENTS:
            gp_density = np.random.uniform(40, 110)     # severly underserved
        elif urban_score == 0:            
            gp_density = np.random.uniform(60, 140)     # rural
        elif urban_score == 1:            
            gp_density = np.random.uniform(90, 170)     # peri-urban
        elif urban_score == 2:            
            gp_density = np.random.uniform(120, 200)    # small city
        else:
            gp_density = np.random.uniform(140, 260)    # Urban

        # Feature 5 -------- gp_count   
        # absolute number of GPs ( derived form density and population)
        # Minimun 0 - some tiny commune shave no GP at all 
        gp_count = max(0, int((gp_density * population)/100_000))

        # Feature 6 -------- specialist density
        # specialists correlate with urban level
        # Rural areas almost never have specialists - people travel 
        specialist_density = gp_density * np.random.uniform(0.3, 1.2)

        # Feature 7 --------- Pharmacy score (1-5)
        # Pharmacists are often the first point of contact
        # in areas without GP
        pharmacy_score = float(np.clip(urban_score * 1.2 + np.random.uniform(-0.5, 1.5), 1, 5))     
        
        # Feature 8 ----------- Wealth Index (0-1)
        # Wealthier areas attract and retain doctors
        # proxy for local government healthcare investment capacity
        if dept in WEALTHY_DEPARTMENTS:
            wealth_index = np.random.uniform(0.65, 1.0)
        elif dept in RURAL_DEPARTMENTS:
            wealth_index = np.random.uniform(0.20, 0.50)
        else: 
            wealth_index = np.random.uniform(0.38, 0.72)

        # Feature 9 ---------- population growth rate
        # declining population = doctors leave and dont return 
        # growing population = attracts services
        if urban_score == 0 and dept in RURAL_DEPARTMENTS:
            pop_growth = np.random.uniform(-0.035, 0.005)
        elif urban_score >= 2:
            pop_growth = np.random.uniform(-0.010, 0.040)
        else:
            pop_growth = np.random.uniform(-0.020, 0.025)

        # Feature 10 ------------ avg_gp_age:
        # Critifcal forward look indicator 
        # if the age of GP is 62, retirements will worsen the crisis
        # Rural Areas have older GPs - Young Doctors prefer Cities
        if urban_score <= 1:
            avg_gp_age = np.random.uniform(50, 64)
        else:
            avg_gp_age = np.random.uniform(44, 58)
        
        # Featue 11 ------------- teleconsult score (1-5)
        # telemedicine can partially compensate for GP shortages
        # requires both infrastructure AND population digital literacy
        teleconsult_score = float(np.clip(urban_score * 1.0 + wealth_index * 2.0 + np.random.uniform(-0.5, 0.5), 1, 5))

        # Target Variable -----------------
        # DREES official definition of medical desert zones
        # GP_Density < 70 % of national average -> desert
        # we add a third class for severe cases (<55%)
        desert_ratio = gp_density / NATIONAL_AVG_GP_DENSITY

        if desert_ratio < 0.55:
            risk_label = 2                          # HIGH - Severe desert
        elif desert_ratio < DESERT_THRESHOLD:
            risk_label = 1                          # medium - emerging desert
        else:
            risk_label = 0                          # LOW - adequate coverage


        records.append({
            # Identifiers (not used in training, kept for lookup)
            "commune_code":       commune.get("code", ""),
            "commune_name":       commune.get("nom", ""),
            "department_code":    dept,
            "region_code":        commune.get("codeRegion", ""),

            # Raw values (for display in dashboard)
            "population":         population,
            "gp_density_per_100k": round(gp_density, 2),

            # ML Features
            "population_log":           round(population_log, 4),
            "urban_score":              urban_score,
            "elderly_ratio":            round(elderly_ratio, 4),
            "gp_count":                 gp_count,
            "specialist_density":       round(specialist_density, 2),
            "pharmacy_score":           round(pharmacy_score, 2),
            "wealth_index":             round(wealth_index, 4),
            "population_growth_rate":   round(pop_growth, 4),
            "avg_gp_age":               round(avg_gp_age, 1),
            "teleconsult_score":        round(teleconsult_score, 2),

            # Target
            "medical_desert_risk":      risk_label,
            "gp_density_ratio":         round(desert_ratio, 4),
        })

    df = pd.DataFrame(records)
    return df

def validate_dataset(df: pd.DataFrame) -> bool:
    """
        Sanity checks before saving.

        In production, data validation is a separate step
        often using libraries like Great Expectations or Pandera.
        Here we do basic checks manually.
    """
    print("Validating Dataset ...............")
    passed = True

    # Check Minimum size
    if len(df) < 100:
        print(f"Dataset too small: {len(df)} rows only")
        passed = False
    
    # Check no nulls in critical columns
    critical = ["population_log", "gp_density_per_100k",
                 "medical_desert_risk", "elderly_ratio"]
    nulls = df[critical].isnull().sum()
    if nulls.any():
        print(f"Null entry found in critical columns {nulls[nulls > 0]}, please fix ")
        passed = False
    
    # check class distribution --  we need all 3 classes 
    classes = df["medical_desert_risk"].unique()
    if len(classes) < 3:
        print(f"Missing Risk classes : found {sorted(classes)}")
        passed = False
    
    # check no negative values where impossible
    if (df["population_log"] < 0).any():
        print("Negative population_log values found ")
        passed = False

    if passed:
        print("Data Successfulyy validated")
    
    return passed

def run_ingestion() -> pd.DataFrame:
    """
        Main entry point.
        Orchestrates fetch → engineer → validate → save.

        Returns the final DataFrame.
    """

    print("=" * 55)
    print("🇫🇷  MedAccès — Data Ingestion Pipeline")
    print("=" * 55)

    # step 1 ------------- fetch
    communes = fetch_communes_from_insee()
    if communes is None:
        communes = build_synthetic_communes()
    
    # Step 2 -------------- Sample if large (keep training fast for now)
    if len(communes) > 3000:
        import random
        random.seed(42)
        communes = random.sample(communes, 3000)
        print("sampled 3000 communes for training")
        
    
    # step 3 -------------- Engineer Features
    df = engineer_features(communes)
    print(f"\n Dataset Shape    : {df.shape}")
    print(f"Class Distribution  :")
    dist = df["medical_desert_risk"].value_counts().sort_index()
    labels = {0: "Low", 1: "Medium", 2: "High"}
    for cls, count in dist.items():
        pct = count / len(df) * 100
        print(f"   {labels[cls]:8s} (class {cls}): {count:4d} ({pct:.1f}%)")

    # step 4 validate -----------------
    if not validate_dataset(df):
         raise ValueError("Dataset validation failed — fix issues before training")

    # step 5 ----------- save 
    output_path = Path(settings.data_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n Dataset save to path : {output_path}")
    print("=" * 55 )

    return df


if __name__ == "__main__":
    df = run_ingestion()

    print("\n Sample Rows (first 5): ")
    print(
         df[[
            "commune_name", "department_code", "population",
            "gp_density_per_100k", "elderly_ratio", "medical_desert_risk"
        ]].head()
        .to_string(index=False)
    )

    print("\nFeature statistics:")
    features = [
        "population_log", "elderly_ratio", "gp_density_per_100k",
        "wealth_index", "avg_gp_age", "teleconsult_score"
    ]
    print(df[features].describe().round(3).to_string())
