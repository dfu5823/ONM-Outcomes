import datetime
import os
import numpy as np
import pandas as pd
import pyreadstat
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split as tts, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from scipy import sparse
from collections import defaultdict
from scipy.sparse import csr_matrix
from sklearn.utils import resample
import dill as pickle
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from functools import reduce
from collections import Counter
from sklearn.model_selection import KFold
from tqdm.auto import tqdm
import xgboost as xgb
import catboost as cb
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from scipy.stats import chi2_contingency
import shap

plt.rcParams.update({'font.size': 16})

# Race distribution plot with updated labels
race_labels = {
    1: 'White',
    2: 'Black',
    3: 'American Indian, Aleutian, or Eskimo',
    4: 'Chinese',
    5: 'Japanese',
    6: 'Filipino',
    7: 'Hawaiian',
    8: 'Korean',
    10: 'Vietnamese',
    11: 'Laotian',
    12: 'Hmong',
    13: 'Kampuchean (incl. Khmer and Cambodian)',
    14: 'Thai',
    15: 'Asian Indian or Pakistani, NOS',
    16: 'Asian Indian',
    17: 'Pakistani',
    20: 'Micronesian, NOS',
    21: 'Chamorran',
    22: 'Guamanian, NOS',
    25: 'Polynesian, NOS',
    26: 'Tahitian',
    27: 'Samoan',
    28: 'Tongan',
    30: 'Melanesian, NOS',
    31: 'Fiji Islander',
    32: 'New Guinean',
    96: 'Other Asian (incl. Asian, NOS and Oriental, NOS)',
    97: 'Pacific Islander, NOS',
    98: 'Other',
    99: 'Unknown'
}

'''
White alone: 76.3%
Black or African American alone: 13.4%
Asian alone: 5.9%
American Indian and Alaska Native alone: 1.3%
Native Hawaiian and Other Pacific Islander alone: 0.2%
Two or More Races: 2.8%
Hispanic or Latino (of any race): 18.5%
'''    
# Insurance status distribution plot with custom labels
insurance_labels = {
    0: 'Not Insured',
    1: 'Private Insurance / Managed Care',
    2: 'Medicaid',
    3: 'Medicare',
    4: 'Other Government',
    9: 'Insurance Status Unknown'
}
# Mapping for MED_INC_QUAR_2016
med_inc_labels = {
    1: '< $40,227',
    2: '$40,227 - $50,353',
    3: '$50,354 - $63,332',
    4: '>=$63,333',
    'Not available': 'Not available'
}

# Mapping for UR_CD_13 to Rural-Urban Grouping
ur_cd_13_labels = {
    1: 'Metro',
    2: 'Metro',
    3: 'Metro',
    4: 'Urban',
    5: 'Urban',
    6: 'Urban',
    7: 'Urban',
    8: 'Rural',
    9: 'Rural',
    'Not available': 'Not available'
}

# Mapping for Tumor Grade
grade_labels = {
    1: 'Grade I', 
    2: 'Grade II', 
    3: 'Grade III', 
    4: 'Grade IV', 
    5: 'T cell', 
    6: 'B Cell', 
    7: 'Non T/B cell', 
    8: 'NK cell', 
    9: 'Other'
}

def visualize_demographics(df):

    # General Statistics
    # Print a concise summary of the DataFrame
    print('General data summary')
    print(df.info())

    # Visualization
    # Histogram for a numerical column
    plt.figure(figsize=(10, 6))
    sns.histplot(df['AGE'], kde=True, bins=30)
    plt.title('Distribution of Age')
    plt.show()
    
    # Visualization
    # Histogram for a numerical column
    plt.figure(figsize=(10, 6))
    sns.histplot(df['REGIONAL_NODES_EXAMINED'], kde=True, bins=30)
    plt.title('Distribution of Regional Lymph Node Exam')
    plt.show()

    # Sex distribution plot
    plt.figure(figsize=(10, 6))
    sns.histplot(df['SEX'], kde=False, discrete=True, bins=[0.5, 1.5, 2.5], stat="density")
    plt.title('Distribution of Sex')
    plt.xticks([1, 2], ['Male', 'Female'])  # Set x-ticks to represent Male and Female
    plt.show()

    # Filter to include only RACE categories with entries
    race_counts = df['RACE'].value_counts()  # This gives you a Series with counts of each unique value
    valid_race_codes = race_counts.index[race_counts > 0]  # Keep only codes with more than 0 entries

    # Update race_labels dictionary to include only valid race codes
    filtered_race_labels = {code: label for code, label in race_labels.items() if code in valid_race_codes}

    # Convert RACE codes to their string labels
    df['RACE_LABEL'] = df['RACE'].map(filtered_race_labels).astype('category')
    race_label_counts = df['RACE_LABEL'].value_counts().index

    plt.figure(figsize=(10, 6))
    sns.countplot(x='RACE_LABEL', data=df[df['RACE_LABEL'].notnull()], order=race_label_counts)
    plt.title('Distribution of Race')
    plt.xticks(rotation=90)
    plt.show()

    # Calculate the percentage of each race category
    race_percentages = df['RACE_LABEL'].value_counts(normalize=True) * 100

    # Print the percentages
    print(race_percentages)

    # Convert INSURANCE_STATUS codes to their string labels
    df['INSURANCE_STATUS_LABEL'] = df['INSURANCE_STATUS'].map(insurance_labels).astype('category')
    insurance_status_label_counts = df['INSURANCE_STATUS_LABEL'].value_counts().index

    plt.figure(figsize=(10, 6))
    sns.countplot(x='INSURANCE_STATUS_LABEL', data=df[df['INSURANCE_STATUS_LABEL'].notnull()], order=insurance_status_label_counts)
    plt.title('Distribution of Insurance Status')
    plt.xticks(rotation=45)
    plt.show()

    # Assuming 'blank' is represented as NaN or a specific marker, map it to 'Not available'
    # If using a marker, replace np.nan with that marker (e.g., -1, or 'blank' if it's a string)
    df['MED_INC_QUAR_2016'] = df['MED_INC_QUAR_2016'].replace(np.nan, 'Not available')

    # First, map the MED_INC_QUAR_2016 values to their labels
    df['MED_INC_QUAR_2016_LABEL'] = df['MED_INC_QUAR_2016'].map(med_inc_labels)

    # Then, convert the column to a categorical type with a defined order
    # Ensure the categories are in the order you want by listing them explicitly
    income_categories = ['< $40,227', '$40,227 - $50,353', '$50,354 - $63,332', '>=$63,333', 'Not available']
    df['MED_INC_QUAR_2016_LABEL'] = pd.Categorical(df['MED_INC_QUAR_2016_LABEL'], categories=income_categories, ordered=True)

    plt.figure(figsize=(10, 6))
    # Now, when you plot, the bars will be in the order of the categories you defined
    sns.histplot(df['MED_INC_QUAR_2016_LABEL'], kde=False)
    plt.title('Distribution of Median Income Quartile 2016')
    plt.xticks(rotation=45)
    plt.show()

    # Assuming 'blank' is represented as NaN or a specific marker, map it to 'Not available'
    # If using a marker, replace np.nan with that marker
    df['UR_CD_13'] = df['UR_CD_13'].replace(np.nan, 'Not available')

    plt.figure(figsize=(10, 6))
    ur_cd_13_plot = sns.histplot(df['UR_CD_13'].map(ur_cd_13_labels), kde=False)
    plt.title('Distribution of Rural Urban Grouping (UR_CD_13)')
    plt.xticks(rotation=45)
    plt.show()
    
    # Assuming 'blank' is represented as NaN or a specific marker, map it to 'Not available'
    # If using a marker, replace np.nan with that marker
    df['GRADE'] = df['GRADE'].replace(np.nan, 'Other')
    
    # First, map the MED_INC_QUAR_2016 values to their labels
    df['GRADE_LABEL'] = df['GRADE'].map(grade_labels)

    # Then, convert the column to a categorical type with a defined order
    # Ensure the categories are in the order you want by listing them explicitly
    grade_categories = ['Grade I', 'Grade II', 'Grade III', 'Grade IV', 'T cell', 'B Cell', 'Non T/B cell', 'NK cell', 'Other']
    df['GRADE_LABEL'] = pd.Categorical(df['GRADE_LABEL'], categories=grade_categories, ordered=True)

    plt.figure(figsize=(10, 6))
    # Now, when you plot, the bars will be in the order of the categories you defined
    sns.histplot(df['GRADE_LABEL'], kde=False)
    plt.title('Distribution of Tumor Grade')
    plt.xticks(rotation=45)
    plt.show()

    print('First 10 rows of data')
    # Temporarily adjust display options
    pd.set_option('display.max_rows', 40)  # Set to 40 or more as needed
    pd.set_option('display.max_columns', None)  # Ensure all columns are shown

    # Display the first 10 rows
    display(df.head(10))

    # Reset display options to default (optional, if you want to revert to standard display settings)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')

def format_summary(series, name):
    return pd.DataFrame({
        name: series.index,
        'Count': series.values,
        'Percentage (%)': (series.values / series.sum() * 100).round(2)
    })

def summarize_demographics(df):
    summaries = []
    
    # Age summary
    age_summary = df['AGE'].describe().round(2)
    
    # Sex summary
    sex_summary = format_summary(df['SEX'].map({1: 'Male', 2: 'Female'}).value_counts(), 'Sex')
    
    # Race summary
    race_summary = format_summary(df['RACE_LABEL'].value_counts(), 'Race')
    
    # Insurance status summary
    insurance_summary = format_summary(df['INSURANCE_STATUS_LABEL'].value_counts(), 'Insurance Status')
    
    # Median income quartile summary
    income_summary = format_summary(df['MED_INC_QUAR_2016_LABEL'].value_counts(), 'Median Income Quartile')
    
    # Urban/rural classification summary
    ur_summary = format_summary(df['UR_CD_13'].map(ur_cd_13_labels).value_counts(), 'Urban/Rural Classification')
    
    # Tumor grade summary
    grade_summary = format_summary(df['GRADE'].map({1: 'Grade I', 2: 'Grade II', 3: 'Grade III', 4: 'Grade IV', 5: 'T cell', 6: 'B Cell', 7: 'Non T/B cell', 8: 'NK cell', 9: 'Other'}).value_counts(), 'Tumor Grade')
    
    # Regional lymph nodes summary
    ln_summary = df['REGIONAL_NODES_EXAMINED'].describe().round(2)
    
    # Aggregate summaries into a dictionary for easy access
    summaries = {
        'Age Summary': age_summary,
        'Sex Distribution': sex_summary,
        'Race Distribution': race_summary,
        'Insurance Status Distribution': insurance_summary,
        'Income Quartile Distribution': income_summary,
        'Urban/Rural Classification': ur_summary,
        'Tumor Grade Distribution': grade_summary,
        'Regional Lymph Node Dissection': ln_summary,
    }
    
    # For Age Summary, since it's not a simple count/percentage, we handle it separately
    age_stats = pd.DataFrame(age_summary).reset_index().rename(columns={'index': 'Statistic', 'AGE': 'Value'})
    
    # For Age Summary, since it's not a simple count/percentage, we handle it separately
    ln_stats = pd.DataFrame(ln_summary).reset_index().rename(columns={'index': 'Statistic', 'REGIONAL_NODES_EXAMINED': 'Value'})
    
    return summaries, age_stats, ln_stats

def filter_columns(df, remove_a_prefix=True, additional_cols_to_remove=None, remove_lowercase=True):
    """
    Filter columns in the dataframe based on whether their names start with 'A_',
    are entirely in lowercase, and/or based on a list of additional column names to remove.

    Parameters:
    - df (pd.DataFrame): The input dataframe from which columns are to be removed.
    - remove_a_prefix (bool): If True, remove columns that start with 'A_'. If False, remove columns that do not start with 'A_'.
    - additional_cols_to_remove (list): List of additional column names to remove from the dataframe.
    - remove_lowercase (bool): If True, remove columns with names that are entirely in lowercase.

    Returns:
    - pd.DataFrame: A dataframe with the specified columns removed.
    """
    # Identify columns that start with 'A_'
    if remove_a_prefix:
        filtered_cols = [col for col in df.columns if not col.startswith('A_')]
    else:
        filtered_cols = [col for col in df.columns if col.startswith('A_')]
    
    # Optionally, remove columns with names that are entirely in lowercase
    if remove_lowercase:
        filtered_cols = [col for col in filtered_cols if not col.islower()]

    # Further remove additional specified columns, if any
    if additional_cols_to_remove is not None:
        filtered_cols = [col for col in filtered_cols if col not in additional_cols_to_remove]

    # Return dataframe with the remaining columns
    return df[filtered_cols]

# Some info on variables including outcomes and variables to remove during preprocessing
# Outcomes:
'''
30 d readmission post-op READM_HOSP_30_DAYS
30 d mortality PUF_30_DAY_MORT_CD
90 d mortality PUF_90_DAY_MORT_CD
Last contact or death, days from dx DX_LASTCONTACT_DEATH_MONTHS
Vital status PUF_VITAL_STATUS
'''
# Variables to exclude to prevent data leakage (occur after the day of surgery)
'''
Variables that begin with "A_"

# Surgical stay duration: 
SURG_DISCHARGE_DAYS

# Reason for no surgery:
REASON_FOR_NO_SURGERY

# 30 d readmission post-op:
READM_HOSP_30_DAYS

# 30 d mortality:
PUF_30_DAY_MORT_CD

# 90 d mortality:
PUF_90_DAY_MORT_CD

# Last contact or death, days from dx:
DX_LASTCONTACT_DEATH_MONTHS

# Vital status:
PUF_VITAL_STATUS

# Treatment summary
RX_SUMM_TREATMENT_STATUS,
DX_RX_STARTED_DAYS,

# Adjuvant radiation related variables (whether they got it or not, what the outcomes were from it)
DX_RAD_STARTED_DAYS, 
RAD_LOCATION_OF_RX, 
PHASE_I_RT_VOLUME, 
PHASE_I_RT_TO_LN, 
PHASE_I_RT_MODALITY, 
PHASE_I_BEAM_TECH, 
PHASE_I_DOSE_FRACT, 
PHASE_I_NUM_FRACT, 
PHASE_I_TOTAL_DOSE, 
PHASE_II_RT_VOLUME, 
PHASE_II_RT_TO_LN, 
PHASE_II_RT_MODALITY, 
PHASE_II_BEAM_TECH, 
PHASE_II_DOSE_FRACT, 
PHASE_II_NUM_FRACT, 
PHASE_II_TOTAL_DOSE, 
PHASE_III_RT_VOLUME,
PHASE_III_RT_TO_LN,
PHASE_III_RT_MODALITY,
PHASE_III_BEAM_TECH,
PHASE_III_DOSE_FRACT,
PHASE_III_NUM_FRACT,
PHASE_III_TOTAL_DOSE,
NUMBER_PHASES_RAD_RX,
RAD_RX_DISC_EARLY,
TOTAL_DOSE,
RX_SUMM_SURGRAD_SEQ,
RAD_ELAPSED_RX_DAYS,
REASON_FOR_NO_RADIATION,

# Adjuvant systemic related variables
DX_SYSTEMIC_STARTED_DAYS,
RX_SUMM_CHEMO,
RX_HOSP_CHEMO,
DX_CHEMO_STARTED_DAYS,
RX_SUMM_HORMONE,
RX_HOSP_HORMONE,
DX_HORMONE_STARTED_DAYS,
RX_SUMM_IMMUNOTHERAPY,
RX_HOSP_IMMUNOTHERAPY,
DX_IMMUNO_STARTED_DAYS,
RX_SUMM_TRNSPLNT_ENDO,
RX_SUMM_SYSTEMIC_SUR_SEQ,

# Other treatment related variables
RX_SUMM_OTHER,
RX_HOSP_OTHER,
DX_OTHER_STARTED_DAYS,
PALLIATIVE_CARE,
PALLIATIVE_CARE_HOSP,
'''

def exclude_variables_by_blank_proportion(df, exclude_variables_more_than_x_proportion_blank=0.5, verbose=False, plot_histogram=False):
    """
    Exclude variables from the dataframe that have more than a specified proportion of blank values.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - exclude_variables_more_than_x_proportion_blank (float): Threshold proportion above which variables are excluded.
    - verbose (bool): If True, print excluded variables and their proportion of blank values.
    - plot_histogram (bool): If True, plot a histogram of the percentage of blank values for variables not entirely filled.

    Returns:
    - pd.DataFrame: A dataframe with variables exceeding the blank threshold excluded.
    """
    proportion_blank = df.isnull().mean()
    columns_to_exclude = proportion_blank[proportion_blank > exclude_variables_more_than_x_proportion_blank].index.tolist()

    if verbose:
        print("Excluded variables and their proportion of blanks:")
        for col in columns_to_exclude:
            print(f"{col}: {proportion_blank[col]:.2%}")

    df_filtered = df.drop(columns=columns_to_exclude)

    if plot_histogram:
        # Exclude 100% filled variables for the histogram
        proportion_blank_filtered = proportion_blank[(proportion_blank > 0) & (proportion_blank <= 1)]
        plt.figure(figsize=(10, 6))
        plt.hist(proportion_blank_filtered, bins=20, color='skyblue', edgecolor='black')
        plt.title('Histogram of Blank Value Proportions')
        plt.xlabel('Proportion Blank')
        plt.ylabel('Number of Variables')
        plt.show()

    return df_filtered

def filter_variables(df,dataset_option,outcome_option,exclude_variables_more_than_x_proportion_blank,verbose=True,plot_histogram=True):

    if dataset_option == "use_full_ncbd_dataset":
        if verbose:
            print('Using full NCBD dataset')
            print('Removing "A_" variables')
        # remove A_ variables
        # select outcomes
        if outcome_option == "use_30_day_readmission":
            outcome_variable = 'READM_HOSP_30_DAYS'
            if verbose:
                print('Outcome is 30 d readmission')
        elif outcome_option == "use_30_day_mortality":
            outcome_variable = 'PUF_30_DAY_MORT_CD'
            if verbose:
                print('Outcome is 30 d mortality')
        elif outcome_option == "use_90_day_mortality":
            outcome_variable = 'PUF_90_DAY_MORT_CD'
            if verbose:
                print('Outcome is 90 d mortality')
        
        # Create outcome_df based on outcome_variable and remove entries with blank or unknown outcomes from the dataset
        outcome_df = pd.DataFrame(df[outcome_variable])

        # Convert blanks to NaN and filter out '9' and NaN in outcome_df
        outcome_df[outcome_variable] = pd.to_numeric(outcome_df[outcome_variable], errors='coerce')  
        # Convert to numeric, invalid parsing will be set as NaN
        valid_outcome_indices = ~outcome_df[outcome_variable].isin([9, np.nan])  # Identify valid indices (not 9 or NaN)

        # Filter both outcome_df and df based on valid_outcome_indices
        outcome_df = outcome_df[valid_outcome_indices]
        df_filtered = df.loc[valid_outcome_indices]  # Correctly use the boolean mask to filter original df

        # Calculate and print the number of entries excluded
        num_excluded_entries = df.shape[0] - df_filtered.shape[0]
        print(f"Number of entries excluded due to missing outcomes: {num_excluded_entries}")
        
        # remove custom columns that start with "A_", plus other treatment related variables
        additional_columns_to_remove = [
             'READM_HOSP_30_DAYS',
             'PUF_30_DAY_MORT_CD',
             'PUF_90_DAY_MORT_CD',
             'DX_LASTCONTACT_DEATH_MONTHS',
             'PUF_VITAL_STATUS',
             'REASON_FOR_NO_SURGERY',
             'RX_SUMM_TREATMENT_STATUS',
             'DX_RX_STARTED_DAYS',
             'DX_RAD_STARTED_DAYS',
             'RAD_LOCATION_OF_RX',
             'PHASE_I_RT_VOLUME',
             'PHASE_I_RT_TO_LN',
             'PHASE_I_RT_MODALITY',
             'PHASE_I_BEAM_TECH',
             'PHASE_I_DOSE_FRACT',
             'PHASE_I_NUM_FRACT',
             'PHASE_I_TOTAL_DOSE',
             'PHASE_II_RT_VOLUME',
             'PHASE_II_RT_TO_LN',
             'PHASE_II_RT_MODALITY',
             'PHASE_II_BEAM_TECH',
             'PHASE_II_DOSE_FRACT',
             'PHASE_II_NUM_FRACT',
             'PHASE_II_TOTAL_DOSE',
             'PHASE_III_RT_VOLUME',
             'PHASE_III_RT_TO_LN',
             'PHASE_III_RT_MODALITY',
             'PHASE_III_BEAM_TECH',
             'PHASE_III_DOSE_FRACT',
             'PHASE_III_NUM_FRACT',
             'PHASE_III_TOTAL_DOSE',
             'NUMBER_PHASES_RAD_RX',
             'RAD_RX_DISC_EARLY',
             'TOTAL_DOSE',
             'RX_SUMM_SURGRAD_SEQ',
             'RAD_ELAPSED_RX_DAYS',
             'REASON_FOR_NO_RADIATION',
             'DX_SYSTEMIC_STARTED_DAYS',
             'RX_SUMM_CHEMO',
             'RX_HOSP_CHEMO',
             'DX_CHEMO_STARTED_DAYS',
             'RX_SUMM_HORMONE',
             'RX_HOSP_HORMONE',
             'DX_HORMONE_STARTED_DAYS',
             'RX_SUMM_IMMUNOTHERAPY',
             'RX_HOSP_IMMUNOTHERAPY',
             'DX_IMMUNO_STARTED_DAYS',
             'RX_SUMM_TRNSPLNT_ENDO',
             'RX_SUMM_SYSTEMIC_SUR_SEQ',
             'RX_SUMM_OTHER',
             'RX_HOSP_OTHER',
             'DX_OTHER_STARTED_DAYS',
             'PALLIATIVE_CARE',
             'PALLIATIVE_CARE_HOSP',
             'PUF_CASE_ID', 
             'PUF_FACILITY_ID',
             'SENTINEL_LNBX_STARTED_DAY',
             'REG_LN_DISS_STARTED_DAY', # contains too many missing values causing issues with imputation
        ]
#         if not outcome_option == "use_30_day_readmission":
#             additional_columns_to_remove.append('SURG_DISCHARGE_DAYS')
        if verbose:
            print('Removing treatment-related variables')
        df_filtered = filter_columns(df_filtered, remove_a_prefix=True, additional_cols_to_remove=additional_columns_to_remove, remove_lowercase=True)

    elif dataset_option == "use_selected_A_dataset":
        if verbose:
            print('Using selected NCBD dataset ("A_" variables)')
            print('Removing non-"A_" variables')
        # remove non A_ variables
        if outcome_option == "use_30_day_readmission":
            outcome_variable = 'A_Readmis'
            if verbose:
                print('Outcome is 30 d readmission')
        elif outcome_option == "use_30_day_mortality":
            outcome_variable = 'A_Mort30day'
            if verbose:
                print('Outcome is 30 d mortality')
        elif outcome_option == "use_90_day_mortality":
            outcome_variable = 'PUF_90_DAY_MORT_CD'
            if verbose:
                print('Outcome is 90 d mortality')
        outcome_df = pd.DataFrame(df[outcome_variable])
        # remove raw columns that do not start with "A_", plus the following variables
        # 'A_Rad', 'A_RadVol', 'A_RadDoseI', 'A_RadDoseII', 'A_RadDoseIII', 'A_RadDoseTot', 'A_RadDose', 'A_Chemo', 'A_Treatment', 'A_SurvivalYrs', 'A_Mort90day', 'A_Mort', 'A_FacType', 'A_FacLoc', 'A_FacVolFreq', 'A_FacVol' 
        additional_columns_to_remove = ['A_Rad', 'A_RadVol', 'A_RadDoseI', 'A_RadDoseII', 'A_RadDoseIII', 'A_RadDoseTot', 'A_RadDose', 'A_Chemo', 'A_Treatment', 'A_SurvivalYrs', 'A_Mort90day', 'A_Mort', 'A_FacType', 'A_FacLoc', 'A_FacVolFreq', 'A_FacVol', 'A_Readmis', 'A_Mort30day']
        if verbose:
            print('Removing treatment-related variables')
        df_filtered = filter_columns(df_filtered, remove_a_prefix=False, additional_cols_to_remove=additional_columns_to_remove, remove_lowercase=True)

    # exclude variables more than exclude_variables_more_than_x_proportion_blank blank
    df_filtered_blanks = exclude_variables_by_blank_proportion(df_filtered, exclude_variables_more_than_x_proportion_blank, verbose=verbose, plot_histogram=plot_histogram)

    print(df_filtered_blanks)
    print(outcome_df)

    return df_filtered_blanks, outcome_df

def custom_train_test_split(X, y, split_ratio, outcome_name='', split_by_dx_date=False):
    """
    Split X and y into training and testing sets based on a given split ratio.
    Always stratifies by the target labels.
    
    Parameters:
    - X (pd.DataFrame): Predictors
    - y (pd.Series or pd.DataFrame): Target
    - split_ratio (list): [train%, test%] (must sum to 100)
    - outcome_name (str): Target column name (needed if split_by_dx_date=True)
    - split_by_dx_date (bool): If True, split chronologically by YEAR_OF_DIAGNOSIS
    
    Returns:
    - X_train, y_train, X_test, y_test
    """
    if len(split_ratio) != 2 or sum(split_ratio) != 100:
        raise ValueError("split_ratio must sum to 100 and contain two elements.")

    test_size = split_ratio[1] / 100.0

    if not split_by_dx_date:
        # Simple stratified split
        X_train, X_test, y_train, y_test = tts(X, y, test_size=test_size, random_state=42, stratify=y)
        return X_train, y_train, X_test, y_test


    # ---- Chronological split with stratification ----
    if 'YEAR_OF_DIAGNOSIS' not in X.columns:
        raise ValueError("'YEAR_OF_DIAGNOSIS' column is required for splitting by diagnosis date.")

    # Combine predictors + target
    combined = pd.concat([X, y.rename(outcome_name)], axis=1)

    # Sort chronologically (breaking ties randomly for reproducibility)
    combined = combined.sample(frac=1, random_state=42).sort_values(
        by='YEAR_OF_DIAGNOSIS', kind='mergesort'
    )

    # Create stratified indices for train/test
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    for train_idx, test_idx in sss.split(combined, combined[outcome_name]):
        train, test = combined.iloc[train_idx], combined.iloc[test_idx]

    X_train, y_train = train.drop(columns=[outcome_name]), train[outcome_name]
    X_test, y_test = test.drop(columns=[outcome_name]), test[outcome_name]

    return X_train, y_train, X_test, y_test

# Function to convert data to numeric, replacing non-numeric placeholders with NaN
def to_numeric(df):
    # Assuming df is a DataFrame; if it's a Series, adjust accordingly
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors='coerce').replace('', np.nan)
    return df

# Function to convert categorical data to string
def to_string(df):
    # Assuming df is a DataFrame; if it's a Series, adjust accordingly
    for column in df.columns:
        df[column] = df[column].astype(str)
    return df
    
def special_impute(df, column, threshold):
    df_copy = df.copy()
    
    # Convert empty strings (or other placeholders for missing data) to NaN
    df_copy[column] = df_copy[column].replace('', np.nan).replace('missing', np.nan)  # Add any other placeholders as needed
    
    # Convert column to numeric, errors='coerce' will turn invalid parsing into NaN
    df_copy[column] = pd.to_numeric(df_copy[column], errors='coerce')
    
    # Now apply the threshold-based NaN marking
    mask = df_copy[column] >= threshold
    df_copy.loc[mask, column] = np.nan
    
    # Calculate the mean of the column excluding NaN values
    mean_value = df_copy[column].mean()
    
    # Replace NaN values with the calculated mean
    df_copy[column] = df_copy[column].fillna(mean_value)
    
    return df_copy

# data normalization:
    # make a dictionary of which variables are continuous vs ordinal vs categorical
        # normalize training continuous variables to be mean centered on 0, std_dev = 1
        # keep ordinal variables as is (but map unknown or NaN to 99 or -1 or 0)
        # one-hot encoding of categorical variables
    # apply the same training normalization to test data

# for each included variable check if there is a corresponding value in these three lists, and group appropriately - if not then group as categorical

# Continuous variables will be normalized to mean=0 std_dev=1 on the training data
continuous_vars = ['AGE', ##### values ==999 should be marked unknown or other and set to mean value
                  'CROWFLY', # distance in mi from nearest hospital
                  'YEAR_OF_DIAGNOSIS',
                  'REGIONAL_NODES_POSITIVE', ##### values >=91 should be marked unknown or other and set to mean value
#                   'REG_LN_DISS_STARTED_DAY', # days between initial dx and LN dissection - excluded due to missing values
#                   'DX_STAGING_PROC_DAYS', # days between initial dx and surgical staging - not present in dataset
                  'TUMOR_SIZE_SUMMARY_16', ##### values >=990 should be marked unknown or other and set to mean value
                  'TUMOR_SIZE',
                  'DX_SURG_STARTED_DAYS', # days between initial dx and first surgical procedure
                  'DX_DEFSURG_STARTED_DAYS', # days between initial dx and definitive surgical resection of primary site
                  'SURG_DISCHARGE_DAYS',
                  'REGIONAL_NODES_EXAMINED', ##### most values are between 0-90, values >=91 should be marked unknown or other and set to mean value
                  ]

# Ordinal variables will be kept as is
ordinal_vars = ['NO_HSD_QUAR_00', # high school degree quartiles
                'NO_HSD_QUAR_12',
                'NO_HSD_QUAR_2016',
                'MED_INC_QUAR_00', # median income quartiles
                'MED_INC_QUAR_12',
                'MED_INC_QUAR_2016',
                'UR_CD_03', # urban-rural index
                'UR_CD_13',
                'CDCC_TOTAL_BEST', # Charlson-Deyo Score 0-max(3)
                'BEHAVIOR',
               ]

# Categorical variables will be one-hot encoded
categorical_vars = ['RACE',
                    'SEX',
                    'FACILITY_TYPE_CD',
                    'FACILITY_LOCATION_CD',
                    'PUF_MULT_SOURCE',
                    'PUF_REFERENCE_DATE_FLAG',
                    'SPANISH_HISPANIC_ORIGIN',
                    'INSURANCE_STATUS',
                    'PUF_MEDICAID_EXPN_CODE',
                    'SEQUENCE_NUMBER', # sequence of malignant/non-malignant tumors in lifetime
                    'CLASS_OF_CASE', # where dx and rx took place
                    'PRIMARY_SITE',
                    'LATERALITY',
                    'GRADE', ##### consider ordinal since values are 1-4 (Grade I to Grade IV), but issue is with coding unknown value 9 (do we encode it to mean or as 5 or 0)
                    'GRADE_CLIN', 
                    'GRADE_PATH', 
                    'GRADE_PATH_POST',
                    'DIAGNOSTIC_CONFIRMATION',
                    'RX_SUMM_DXSTG_PROC',
                    'RX_HOSP_DXSTG_PROC',
                    'TNM_CLIN_T',
                    'TNM_CLIN_N',
                    'TNM_CLIN_M',
                    'TNM_CLIN_STAGE_GROUP',
                    'TNM_PATH_T',
                    'TNM_PATH_N',
                    'TNM_PATH_M',
                    'TNM_PATH_STAGE_GROUP',
                    'TNM_EDITION_NUMBER',
                    'ANALYTIC_STAGE_GROUP',
#                     'METS_AT_DX', ##### all the values that contain this string should be categorical
#                     'AJCC_TNM', ##### all the values that contain this string should be categorical
#                     'CS_SITESPECIFIC_FACTOR', ##### all the values that contain this string should be categorical
                    'CS_VERSION_LATEST', 
                    'CS_EXTENSION',
                    'CS_TUMOR_SIZEEXT_EVAL',
                    'LYMPH_VASCULAR_INVASION',
#                     'CS_METS', ##### all the values that contain this string should be categorical
                    'RX_SUMM_SURG_PRIM_SITE',
                    'RX_HOSP_SURG_PRIM_SITE',
                    'RX_HOSP_SURG_APPR_2010',
                    'RX_SUMM_SURGICAL_MARGINS',
                    'RX_SUMM_SCOPE_REG_LN_SUR',
                    'RX_SUMM_SCOPE_REG_LN_2012',
                    'RX_SUMM_SURG_OTH_REGDIS',
                   ]
    
def normalize(X_train, X_test):
    # Drop any columns with missing values
    cols_to_remove = [col for col in X_train.columns if X_train[col].isna().all()]
    X_train = X_train.drop(columns=cols_to_remove)
    X_test = X_test.drop(columns=cols_to_remove)
    
    # Handle special imputation cases for continuous variables
    special_cases = {
        'AGE': 999,
        'REGIONAL_NODES_POSITIVE': 91,
        'TUMOR_SIZE_SUMMARY_16': 990,
        'REGIONAL_NODES_EXAMINED': 91
    }
    for var, threshold in special_cases.items():
        if var in continuous_vars:
            X_train = special_impute(X_train, var, threshold)
            X_test = special_impute(X_test, var, threshold)  # Apply the same mean from train

    # Identify all columns in X_train that are not explicitly classified
    all_vars = set(X_train.columns)
    specified_vars = set(continuous_vars + ordinal_vars + categorical_vars)
    unclassified_vars = list(all_vars - specified_vars)
        
    # Treat unclassified variables as categorical
    extended_categorical_vars = categorical_vars + unclassified_vars
    
    # Create a FunctionTransformer to use the string '' to float NaN converter in the continuous pipeline
    numeric_transformer = FunctionTransformer(to_numeric)
    
    # Create a FunctionTransformer to use the float to string converter in the categorical pipeline
    string_transformer = FunctionTransformer(to_string)

    continuous_pipeline = Pipeline(steps=[
        ('to_numeric', numeric_transformer),  # Convert to numeric and handle non-numeric as NaN
        ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with the mean
        ('scaler', StandardScaler())  # Scale the data
    ])
    
    ordinal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ])
    
    categorical_pipeline = Pipeline(steps=[
        ('to_string', string_transformer),  # Convert all data to string
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Handle missing values
        ('encoder', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode the categorical data
    ])
    
    # ColumnTransformer now includes extended_categorical_vars
    preprocessor = ColumnTransformer([
        ('continuous', continuous_pipeline, continuous_vars),
        ('ordinal', ordinal_pipeline, ordinal_vars),
        ('categorical', categorical_pipeline, extended_categorical_vars)
    ], remainder='passthrough') 
    
    # Fit the preprocessor on X_train
    preprocessor.fit(X_train)
    
    # Transform X_train and X_test
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # If the output is sparse, convert it to a dense format
    if sparse.issparse(X_train_transformed):
        X_train_transformed = X_train_transformed.toarray()
    if sparse.issparse(X_test_transformed):
        X_test_transformed = X_test_transformed.toarray()
        
    # Generate new column names from the transformers
    new_columns = (
        continuous_vars +
        ordinal_vars + preprocessor.named_transformers_['categorical'].named_steps['encoder'].get_feature_names_out(extended_categorical_vars).tolist()
    )
    
    # Convert transformed data back to DataFrame
    X_train_df = pd.DataFrame(X_train_transformed, columns=new_columns, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_transformed, columns=new_columns, index=X_test.index)
    
    return X_train_df, X_test_df
    

'''
30 Day Readmission coding
READM_HOSP_30_DAYS
0: No surgical procedure of the primary site was performed, or the patient was not readmitted to the same hospital within 30 days of discharge
1: A patient was surgically treated and was readmitted to the same hospital within 30 days of being discharged. This readmission was unplanned
2: A patient was surgically treated and was then readmitted to the same hospital within 30 days of being discharged. This readmission was planned (chemotherapy port insertion, revision of colostomy, etc.)
3: A patient was surgically treated and, within 30 days of being discharged, the patient had both a planned and an unplanned readmission to the same hospital
9: It is unknown whether surgery of the primary site was recommended or performed. It is unknown whether the patient was readmitted to the same hospital within 30 days of discharge
'''

'''
30 (or 90) Day Mortality coding
PUF_30_DAY_MORT_CD (PUF_90_DAY_MORT_CD)
0: Patient alive, or died more than 30 (90) days after surgery performed
1: Patient died <= 30 (90) days from surgery date
9: Patient alive with fewer than 30 (90) days of follow-up, surgery date missing, or last contact date missing
blank: Not eligible; surgical resection unknown or not performed, or diagnosed in 2019
'''

# Outcome binarization:
#     Values of 0 will stay as 0
#     Values >= 1 will become 1
#     Note that values of 9 or blank were already removed in the filter_variables function

def binarize(y_train_raw, y_test_raw):
    """
    Binarize outcome variables.
    
    Parameters:
    - y_train_raw: Raw training target values.
    - y_test_raw: Raw testing target values.
    
    Returns:
    - y_train: Binarized training target values.
    - y_test: Binarized testing target values.
    """
    
    # Binarization process
    y_train = np.where(y_train_raw >= 1, 1, y_train_raw).ravel()
    y_test = np.where(y_test_raw >= 1, 1, y_test_raw).ravel()
    
    return y_train, y_test

def get_overlapping_indices(*index_arrays, overlap_mode="all_overlap"):
        """
        Get indices that overlap in multiple arrays.

        Parameters:
        *index_arrays (multiple np.arrays): Arrays containing indices.
        overlap_mode (str): Determines the type of overlap to look for in the indices.
                            "all_overlap" for indices common to all arrays,
                            "two_overlap" for indices appearing in at least two arrays.

        Returns:
        np.array: An array of indices that meet the specified overlap condition.
        """

        # Filter out empty arrays and ensure all inputs are arrays (for safety)
        arrays = [np.array(x) for x in index_arrays if len(x) != 0]

        if overlap_mode == "all_overlap":
            # Compute the common indices from all methods
            if arrays:
                common_indices = reduce(np.intersect1d, arrays)
            else:
                common_indices = np.array([])  # Return empty if no arrays provided
            return common_indices

        elif overlap_mode == "two_overlap":
            # Flatten the list of arrays and count the occurrence of each index
            index_counter = Counter(np.concatenate(arrays))

            # Select only the indices that occur more than once
            overlapping_indices = [index for index, count in index_counter.items() if count > 1]

            # Convert to a NumPy array (if you need the result as an array)
            overlapping_indices = np.array(overlapping_indices)
            return overlapping_indices

        else:
            raise ValueError("Invalid overlap_mode. Accepted values are 'all_overlap' or 'two_overlap'.")

import re
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from group_lasso import GroupLasso
def feature_reduction(use_lasso, use_rfe, use_rf, use_boruta,
                      X_train, X_test, y_train, y_test, overlap_mode="all_overlap"):
    """
    Group-aware feature reduction with target <50 groups.
      - use_lasso: Group Lasso
      - use_rfe:   Sparse Group Lasso (group lasso + small L1)
      - use_rf:    Grouped Random Forest
    Auto-detects groups from column names and returns X_train, X_test restricted to selected groups.
    """

    # --- knobs you can tweak ---
    TARGET_GROUPS = 50        # final number cap
    MIN_GROUPS    = 10        # don't go below this
    ALPHA_GL      = 0.05      # group penalty strength
    L1_RATIO_SGL  = 0.15      # >0 makes it sparse group lasso
    RF_N_EST      = 800
    RF_CUM_THR    = 0.50      # cumulative importance threshold (before top-K fallback)
    # ---------------------------

    def _ensure_df(X):
        return X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

    def _infer_group(col: str) -> str:
        c = re.sub(r"\s+", "_", str(col)).replace(".", "_").strip()
        if "_" not in c: return c
        parts = c.split("_")
        last = parts[-1].lower()
        if re.fullmatch(r"(?:p|c)?[0-9a-z]+", last) or last == "nan":
            base = "_".join(parts[:-1])
            return base if base else c
        return "_".join(parts[:-1]) if len(parts) > 1 else c

    def _build_groups(cols):
        groups = {}
        for i, c in enumerate(cols):
            g = _infer_group(c)
            groups.setdefault(g, []).append(i)
        return groups  # {group_name: [col_idx,...]}

    def _group_lasso_scores(Xdf, y, groups, alpha, l1_ratio):
        gnames = list(groups.keys())
        gassign = np.zeros(Xdf.shape[1], dtype=int)
        for gi, g in enumerate(gnames):
            for j in groups[g]: gassign[j] = gi
        Xs = StandardScaler().fit_transform(Xdf.values)
        gl = GroupLasso(groups=gassign, group_reg=alpha, l1_reg=alpha*l1_ratio,
                        frobenius_lipschitz=True, scale_reg="group_size",
                        supress_warning=True, n_iter=2000, tol=1e-4, fit_intercept=True)
        gl.fit(Xs, y)
        b = gl.coef_.ravel()
        return {g: float(np.linalg.norm(b[groups[g]], 2)) for g in gnames}

    def _rf_group_importance(Xdf, y, groups):
        rf = RandomForestClassifier(n_estimators=RF_N_EST, random_state=42, n_jobs=-1)
        rf.fit(Xdf, y)
        fi = rf.feature_importances_
        return {g: float(fi[groups[g]].sum()) for g in groups}

    def _normalize(d):
        if not d: return d
        vals = np.array(list(d.values()), dtype=float)
        m, M = vals.min(), vals.max()
        if M <= 0 or np.isclose(M, m):
            return {k: 0.0 for k in d}
        return {k: float((v - m) / (M - m)) for k, v in d.items()}

    def _rank_topk(score_dict, topk):
        ordered = sorted(score_dict.items(), key=lambda kv: kv[1], reverse=True)
        return [g for g, _ in ordered[:topk]]

    def _flatten_indices(groups, chosen_groups):
        keep = []
        for g in chosen_groups:
            keep.extend(groups[g])
        return sorted(set(keep))

    # ---- main ----
    Xtr, Xte = _ensure_df(X_train), _ensure_df(X_test)
    print(f"\nThere are {Xtr.shape[1]} features in the training data.")
    groups = _build_groups(Xtr.columns)
    print(f"Auto-detected {len(groups)} variable groups.")

    if not (use_lasso or use_rfe or use_rf):
        print("\nNo methods selected; returning original matrices.")
        return X_train, X_test

    # 1) per-method group scores
    method_scores = []
    hard_sets = []

    if use_lasso:
        s_gl = _normalize(_group_lasso_scores(Xtr, y_train, groups, alpha=ALPHA_GL, l1_ratio=0.0))
        method_scores.append(s_gl)
        hard_sets.append({g for g, v in s_gl.items() if v > 0})

    if use_rfe:
        s_sgl = _normalize(_group_lasso_scores(Xtr, y_train, groups, alpha=ALPHA_GL, l1_ratio=L1_RATIO_SGL))
        method_scores.append(s_sgl)
        hard_sets.append({g for g, v in s_sgl.items() if v > 0})

    if use_rf:
        s_rf = _normalize(_rf_group_importance(Xtr, y_train, groups))
        if any(v > 0 for v in s_rf.values()):
            ordered = sorted(s_rf.items(), key=lambda kv: kv[1], reverse=True)
            cum, keep = 0.0, []
            for g, v in ordered:
                cum += v
                keep.append(g)
                if cum >= RF_CUM_THR: break
            hard_sets.append(set(keep))
        method_scores.append(s_rf)

    # 2) strict overlap if requested
    if hard_sets:
        if overlap_mode == "all_overlap":
            hard = set.intersection(*hard_sets) if all(hard_sets) else set()
        else:
            hard = set.union(*hard_sets)
    else:
        hard = set()

    # 3) consensus score (mean of normalized scores)
    consensus = defaultdict(float)
    counts = defaultdict(int)
    for sd in method_scores:
        for g, v in sd.items():
            consensus[g] += v
            counts[g] += 1
    for g in list(consensus.keys()):
        consensus[g] /= max(1, counts[g])

    # 4) choose final groups
    if hard:
        hard_ranked = _rank_topk({g: consensus.get(g, 0.0) for g in hard}, topk=max(TARGET_GROUPS, MIN_GROUPS))
        chosen = hard_ranked[:TARGET_GROUPS]
    else:
        chosen = _rank_topk(consensus, topk=max(TARGET_GROUPS, MIN_GROUPS))

    # --- force-keep group(s): ensure CLIN_STAGE columns are always included ---
    FORCE_GROUP_PREFIXES = ["TNM_CLIN_STAGE_GROUP"]  # add more prefixes if needed

    def _force_groups(groups, cols, prefixes):
        forced = set()
        for g, idxs in groups.items():
            for i in idxs:
                cname = str(cols[i]).replace(" ", "")
                if any(cname.startswith(p.replace(" ", "")) for p in prefixes):
                    forced.add(g)
                    break
        return forced

    forced_groups = _force_groups(groups, Xtr.columns, FORCE_GROUP_PREFIXES)

    if forced_groups:
        ordered = list(dict.fromkeys(list(forced_groups) + chosen))
        if len(ordered) > TARGET_GROUPS:
            keep_head = list(forced_groups)
            remaining_slots = max(TARGET_GROUPS - len(keep_head), 0)
            rest = [g for g in ordered if g not in forced_groups][:remaining_slots]
            chosen = keep_head + rest
        else:
            chosen = ordered
    # -------------------------------------------------------------------------

    print(f'\nSelected {len(chosen)} groups (target {TARGET_GROUPS}).')

    keep_idx = _flatten_indices(groups, chosen)
    Xtr_sel = Xtr.iloc[:, keep_idx]
    Xte_sel = Xte.iloc[:, keep_idx]

    if isinstance(X_train, pd.DataFrame):
        return Xtr_sel, Xte_sel
    else:
        return Xtr_sel.values, Xte_sel.values



# def feature_reduction(use_lasso,use_rfe,use_rf,use_boruta,X_train,X_test,y_train,y_test,overlap_mode="all_overlap"):
#     print('\n')
#     print(f'There are {X_train.shape[1]} features in the training data.')

#     if use_lasso or use_rfe or use_rf or use_boruta:
#         lasso_support_indices = []
#         rfe_support_indices = []
#         rf_support_indices = []
#         boruta_support_indices = []

#         # Use Lasso for feature reduction (only use features with non-zero coefficients)
#         if use_lasso:
#             cv = KFold(n_splits=5, shuffle=True, random_state=42)
#             lasso = LassoCV(cv=cv).fit(X_train, y_train)
#             coef = lasso.coef_
#             important_features_indices = np.where(coef != 0)[0]
#             # If X_train is a pandas DataFrame
#             if isinstance(X_train, pd.DataFrame):
#                 X_lasso = X_train.iloc[:, important_features_indices]
#             else:  # Assuming X_train is a numpy array
#                 X_lasso = X_train[:, important_features_indices]
#             lasso_support_indices = np.where(lasso.coef_ != 0)[0]

#             print('\n')
#             print(f'Lasso regression selected {X_lasso.shape[1]} important features in the training data.')

#         # Use Recursive Feature Elimination for feature reduction to a pre-set number of genes
#         if use_rfe:
#             model = LinearRegression()
#             number_of_rfe_genes_to_keep = 100
#             rfe = RFE(model, n_features_to_select=number_of_rfe_genes_to_keep)
#             rfe.fit(X_train, y_train)
#             X_rfe = X_train[:, rfe.support_]
#             rfe_support_indices = np.where(rfe.support_)[0]

#             print('\n')
#             print(f'Recursive feature elimination (RFE) eliminated  all but {X_rfe.shape[1]} important features in the training data.')

#         # Use Random Forest for feature reduction (select the n most important genes)
#         if use_rf:
#             total_rf_feature_importance = 0.5
#             rf = RandomForestClassifier(n_estimators=1000, random_state=42)
#             rf.fit(X_train, y_train)

#             # Get feature importances and calculate the cumulative sum
#             importances = rf.feature_importances_
#             cumsum_importances = np.cumsum(np.sort(importances)[::-1])

#             # Get the indices that would sort the importances array
#             sorted_indices = np.argsort(importances)[::-1]

#             # Get the indices for the features that make up 50% of the importance
#             rf_support_indices = sorted_indices[:np.where(cumsum_importances > total_rf_feature_importance)[0][0]]

#             print(f'\nRandom Forest selected {len(rf_support_indices)} important features in the training data with total {total_rf_feature_importance*100}% feature importance.')

#         # Use Boruta for feature reduction (select the n most important genes)
#         if use_boruta:
#             z_score_threshold = 50
#             rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, random_state=42)
#             boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=42, perc=z_score_threshold)   
#             boruta_selector.fit(X_train, y_train)

#             boruta_support_indices = np.where(boruta_selector.support_)[0]

#             print(f'\nBoruta selected {len(boruta_support_indices)} important features in the training data at {z_score_threshold} z-score threshold difference.')

#         common_indices = get_overlapping_indices(rfe_support_indices, lasso_support_indices, rf_support_indices, boruta_support_indices, overlap_mode=overlap_mode)

#         print('\n')
#         print(f'After combining overlapping reduced features from multiple methods, {len(common_indices)} features remain.')

#         if isinstance(X_train, pd.DataFrame):
#             X_train = X_train.iloc[:, common_indices]
#             X_test = X_test.iloc[:, common_indices]
#         else:
#             X_train = X_train[:, common_indices]
#             X_test = X_test[:, common_indices]
#         print('\n')
#         print(f'X_train and X_test now have {X_train.shape[1]} features each from the feature reduction.')
        
#         return X_train,X_test

def pca_transform(X_train, X_test, number_of_PCA_components):
    # Initialize PCA with the number of components and a random state for reproducibility
    pca = PCA(n_components=number_of_PCA_components, random_state=42)
    
    # Fit PCA on the training data and transform both training and test data
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    # Create column names for each PCA component
    columns = [f'PCA_component_{i+1}' for i in range(number_of_PCA_components)]
    
    # Convert the numpy arrays back to pandas dataframe, with the new column names
    X_train_df = pd.DataFrame(X_train_pca, columns=columns, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_pca, columns=columns, index=X_test.index)
    
    # Printing the number of features is no longer necessary as the DataFrame will show it
    # but it's included here for completeness
    print(f'There are {X_train_df.shape[1]} features in the training data after PCA.')
    
    return X_train_df, X_test_df

def SMOTE_resample(X_train, X_test, y_train, y_test, sampling_strategy=0.2):
    print('\n')
    print(f'There are {y_train.sum()} or {round(y_train.sum()/len(y_train)*100,2)}% positive examples of out {len(y_train)} samples in the training data.')
    print(f'There are {y_test.sum()} or {round(y_test.sum()/len(y_test)*100,2)}% positive examples of out {len(y_test)} samples in the test data.')

    # Save the X-train and y-train varaiables for train dataset surival curves below
    X_train_no_SMOTE = X_train
    y_train_no_SMOTE = y_train

    # Instantiate the SMOTE algorithm and resample the data
    smote = SMOTE(sampling_strategy=sampling_strategy,random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    print('\n')
    print(f'After SMOTE there are {y_train.sum()} or {round(y_train.sum()/len(y_train)*100)}% positive examples in the training data out of {len(y_train)} samples in the training data.')
    return X_train,y_train

def plot_covariance_heatmap(X_train, y_train, top_n=20):
    # Ensure y_train is properly formatted
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]  # Assuming y_train is a DataFrame with one column
    elif isinstance(y_train, np.ndarray):
        y_train = pd.Series(y_train.ravel(), name='Outcome')  # Flatten y_train to 1D
    
    # Convert X_train from csr_matrix to DataFrame if it's sparse
    if sparse.issparse(X_train):
        X_train_df = pd.DataFrame(X_train.toarray(), columns=X_train.columns)
    else:
        # Assuming X_train is already a DataFrame, directly use it
        X_train_df = X_train
    
    # Calculate correlation of each feature with the outcome
    correlations = X_train_df.apply(lambda x: y_train.corr(x)).abs()
    
    # Select top_n features based on their absolute correlation with the outcome
    top_features = correlations.sort_values(ascending=False).head(top_n).index.tolist()
    
    # Ensure 'Outcome' is at the beginning of the list
    top_features = ['Outcome'] + [feat for feat in top_features if feat != 'Outcome']
    
    # Recombine X_train_df with y_train for correlation calculation
    combined_df = pd.concat([X_train_df, y_train], axis=1)
    
    # Calculate the correlation matrix for the top features including the outcome
    top_corr_matrix = combined_df[top_features].corr()
    
    # Plot the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(top_corr_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title(f'Top {top_n} Variables with Greatest Correlation to the Outcome')
    plt.show()

# Adjust the model fitting and scoring part to use roc_auc_score and model's probability output
def model_fit_score(model, X_train, y_train, X_val, y_val, param_set):
    model.set_params(**param_set)
    model.fit(X_train, y_train)
    
    # Predict probabilities for the positive class (index 1)
    if hasattr(model, "predict_proba"):
        y_pred_probs = model.predict_proba(X_val)[:, 1]
    else:  # For neural network model
        y_pred_probs = model.predict(X_val).ravel()
    
    # Calculate ROC-AUC score
    score = roc_auc_score(y_val, y_pred_probs)
    return score

def create_keras_classifier_fn(input_dim, batch_size=32, epochs=10):
    def model_fn():
        model = Sequential()
        model.add(Dense(12, input_dim=input_dim, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    return KerasClassifier(build_fn=model_fn, epochs=epochs, batch_size=batch_size, verbose=0, random_state=42)

def make_xgb():
    m = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        tree_method="hist",
        enable_categorical=False,
    )
    # Some xgboost builds expose `feature_weights` in __init__ signature,
    # which makes sklearn's get_params() try to getattr(self, 'feature_weights').
    # Ensure it exists to avoid AttributeError during fit().
    if not hasattr(m, "feature_weights"):
        m.feature_weights = None
    return m

models_hyperparams = {
    "XGBoost": {
        "model": make_xgb,
        "params_ranges": [
            {"n_estimators": n, "max_depth": d, "learning_rate": lr}
            for n in [100, 200]
            for d in [3, 5]
            for lr in [0.01, 0.1]
        ],
    },
    'CatBoost': {
        'model': lambda: cb.CatBoostClassifier(verbose=0, random_state=42),
        'params_ranges': [{'iterations': n, 'depth': d, 'learning_rate': lr} for n in [100, 200] for d in [3, 5] for lr in [0.01, 0.1]]
    },
    'Logistic Regression': {
        'model': lambda: LogisticRegression(random_state=42, max_iter=1000),
        'params_ranges': [
            {'C': c, 'penalty': p}
            for c in [0.01, 0.1, 1, 10, 100]
            for p in ['l2']
        ]
    },
#     'SVC': {
#         'model': lambda: SVC(kernel='linear', probability=True, random_state=42),
#         'params_ranges': [{'C': c} for c in [0.01, 0.1, 1, 10, 100]]
#     },
    'SGD Logistic Regression': {
        'model': lambda: SGDClassifier(loss='log_loss', random_state=42, max_iter=1000, tol=1e-3),
        'params_ranges': [{'penalty': p} for p in ['l2', 'l1', 'elasticnet']]
    },
    'Gradient Boosting': {
        'model': lambda: GradientBoostingClassifier(random_state=42),
        'params_ranges': [{'n_estimators': n, 'learning_rate': lr, 'max_depth': md} for n in [100, 200] for lr in [0.01, 0.1] for md in [3, 5]]
    },
#     'LightGBM': {
#         'model': lambda: lgb.LGBMClassifier(random_state=42),
#         'params_ranges': [{'num_leaves': nl, 'max_depth': md, 'learning_rate': lr} for nl in [31, 63] for md in [-1, 5] for lr in [0.01, 0.1]]
#     },
#     'Neural Networks': {
#         'model': create_keras_classifier_fn,
#         'params_ranges': [{'batch_size': bs, 'epochs': e} for bs in [32, 64] for e in [1, 2]]  # Adjust epochs for quick testing
#     },
}

def bootstrap_nested_cv(models_hyperparams, X, y, num_bootstrap_iterations=100, num_splits=5):
    # Results dictionary to store model performance
    results = defaultdict(lambda: {
        'param_scores': defaultdict(list),  # This ensures param_scores is a dict of lists
        'best_params': [],
        'best_scores': []
    })
    all_models = {}

    for model_name, model_info in models_hyperparams.items():
        params_ranges = model_info['params_ranges']
        
        for bootstrap_iteration in tqdm(range(num_bootstrap_iterations), desc=f'Training {model_name}'):
            X_resampled, y_resampled = resample(X, y, random_state=42)
            if bootstrap_iteration == 1:
                X_resampled = X; y_resampled = y
            best_score = -np.inf
            best_params = None
            
            for param_set in tqdm(params_ranges):
                scores = []
                for train_index, test_index in KFold(n_splits=num_splits, shuffle=True, random_state=42).split(X_resampled):
                    X_train, X_val = X_resampled.iloc[train_index], X_resampled.iloc[test_index]
                    y_train, y_val = y_resampled[train_index].ravel(), y_resampled[test_index].ravel()

                    # Create a fresh model instance for each parameter set and fold
                    fresh_model = model_info['model']()
                    if model_name != 'Neural Networks': # Adjust model instantiation for Keras
                        fresh_model.set_params(**param_set)  # Apply parameters normally
                    score = model_fit_score(fresh_model, X_train, y_train, X_val, y_val, param_set)
                    scores.append(score)
                
                mean_score = np.mean(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = param_set

                results[model_name]['param_scores'][str(param_set)].append(mean_score)
            
            results[model_name]['best_params'].append(best_params)
            results[model_name]['best_scores'].append(best_score)

        # Summarizing and printing the best parameter set for each model
        best_params_overall = max(results[model_name]['best_params'], key=lambda x: np.mean(results[model_name]['param_scores'][str(x)]))
        print(f"Best parameters for {model_name}: {best_params_overall}")
        if model_name == 'Neural Networks':
            # Directly use the function for Neural Networks to handle its unique parameter setting
            all_models[model_name] = model_info['model'](**best_params_overall)
        else:
            # For other models, instantiate a fresh model and then apply best parameters
            fresh_model = model_info['model']()  # Create a new instance
            fresh_model.set_params(**best_params_overall)  # Apply the best parameters
            all_models[model_name] = fresh_model

    return results, all_models

def process_and_save_combined(model_name, model_hyperparams, X_train, y_train, n_bootstraps_for_training, num_k_fold_splits, experiment_name=""):
    # Process the model
    results, model = bootstrap_nested_cv({model_name: model_hyperparams},
                                         X_train, y_train, 
                                         num_bootstrap_iterations=n_bootstraps_for_training, num_splits=num_k_fold_splits)
    
    # Save results and model combined
    save_filename = f'models/{experiment_name}_{model_name}_results_and_cv_model.pkl'
    with open(save_filename, 'wb') as file:
        pickle.dump({"results": results, "model": model}, file)
    
    return results, model

def stack_models(all_models):
    """Stacks multiple models. Here, all_models is expected to be a dictionary
    with model names as keys and model instances as values."""
    estimators = [(name, model) for name, model in all_models.items()]
    stacked_model = StackingClassifier(estimators=estimators, final_estimator=None, cv=5, stack_method='auto', n_jobs=1)
    return stacked_model

def retrain_models(stacked_model, all_models, X_train, y_train):
    """Retrains all models with optimal parameters on the full training dataset."""
    for model_name, model in all_models.items():
        model.fit(X_train, y_train)  # Retrain each model
    stacked_model.fit(X_train, y_train)  # Retrain stacked model
    return stacked_model, all_models

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

def calibrate_models(
    trained_stacked_model,
    trained_all_models: Dict[str, object],
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    method: str = "auto",
    iso_min_pos: int = 100,
) -> Tuple[object, Dict[str, object]]:
    """
    Calibrate (1) each base model in trained_all_models and (2) the final stacked predictor.

    Parameters
    ----------
    trained_stacked_model : estimator
        Either an sklearn StackingClassifier OR your meta-model used with base probs.
    trained_all_models : dict[str, estimator]
        Dict of prefit base estimators (must implement predict_proba).
        NOTE: These calibrated wrappers are for standalone use. If your stacker was
        trained on RAW base probs, do NOT feed calibrated base outputs to it unless
        you retrain the stacker on calibrated base probs.
    X_val, y_val : validation set
        Must be at NATURAL prevalence (no SMOTE) for proper calibration.
    method : {"auto", "sigmoid", "isotonic"}
        - "auto": uses "sigmoid" (Platt) when positives < iso_min_pos, else "isotonic".
        - "sigmoid": always Platt scaling.
        - "isotonic": always isotonic regression.
    iso_min_pos : int
        Minimum #positives in y_val to allow isotonic; otherwise fallback to sigmoid.

    Returns
    -------
    calibrated_stacked_model : estimator-like
        If original is StackingClassifier: a CalibratedClassifierCV(cv="prefit").
        Else: a lightweight wrapper with predict_proba that applies a learned
        1D calibrator on the stacker’s raw probability.
    calibrated_all_models : dict[str, estimator]
        Each base model wrapped with CalibratedClassifierCV(cv="prefit").
    """
    y_val = np.asarray(y_val).ravel()
    n_pos = int(y_val.sum())

    if method not in {"auto", "sigmoid", "isotonic"}:
        raise ValueError("method must be one of {'auto','sigmoid','isotonic'}")

    chosen = "sigmoid"
    if method == "isotonic":
        chosen = "isotonic"
    elif method == "auto":
        chosen = "isotonic" if n_pos >= iso_min_pos else "sigmoid"
    else:  # "sigmoid"
        chosen = "sigmoid"

    # --- 1) Calibrate each base model (for standalone use/inspection) ---
    calibrated_all = {}
    for name, base in trained_all_models.items():
        cal = CalibratedClassifierCV(base, method=chosen, cv="prefit")
        cal.fit(X_val, y_val)
        calibrated_all[name] = cal

    # --- 2) Calibrate final stack ---
    # Directly wrap the prefit StackingClassifier
    cal_stack = CalibratedClassifierCV(trained_stacked_model, method=chosen, cv="prefit")
    cal_stack.fit(X_val, y_val)
    calibrated_stacker = cal_stack

    return calibrated_stacker, calibrated_all

def plot_confusion_matrix(conf_matrix, class_labels, title='Confusion Matrix'):
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots()
    percentage_matrix = conf_matrix / np.sum(conf_matrix) * 100
    sns.heatmap(conf_matrix, annot=False, fmt="d", cmap="coolwarm", ax=ax, cbar=True,
                xticklabels=class_labels, yticklabels=class_labels)

    # Iterate over data to create text annotations with percentages.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            percentage = percentage_matrix[i, j]
            text = f"{conf_matrix[i, j]}\n({percentage:.1f}%)"
            ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", color="black")

    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.xticks(rotation=0)
    plt.title(title)
    plt.show()
    plt.rcParams.update({'font.size': 16})

def bootstrap_confidence_interval(y_true, y_pred, score_func, n_bootstraps, rng_seed=42):
    rng = np.random.RandomState(rng_seed)
    bootstrapped_scores = []
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # Prevent cases where the sample does not include both classes
            continue
        score = score_func(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
    ci_lower, ci_upper = np.percentile(bootstrapped_scores, [2.5, 97.5])
    return np.mean(bootstrapped_scores), ci_lower, ci_upper

def print_metrics(y_true, y_pred_proba, threshold, model_name, class_labels, n_bootstraps):
    y_pred = (y_pred_proba >= threshold).astype(int)
    metrics = {
        'Accuracy': accuracy_score,
        'Sensitivity': lambda y_true, y_pred: recall_score(y_true, y_pred, pos_label=1),
        'Specificity': lambda y_true, y_pred: recall_score(y_true, y_pred, pos_label=0),
        'F1-Score': f1_score
    }
    
    # Plot confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(conf_matrix, class_labels, title=f'Confusion Matrix for {model_name}')
    
    print(f"\n{model_name}:")
    mean_score, ci_lower, ci_upper = bootstrap_confidence_interval(y_true, y_pred_proba, roc_auc_score, n_bootstraps)
    print(f"AUC: {mean_score:.2f} (95% CI: {ci_lower:.2f}-{ci_upper:.2f})")
    for metric_name, metric_func in metrics.items():
        mean_score, ci_lower, ci_upper = bootstrap_confidence_interval(y_true, y_pred, metric_func, n_bootstraps)
        print(f"{metric_name}: {mean_score:.2f} (95% CI: {ci_lower:.2f}-{ci_upper:.2f})")
    
    # Separating into high and low risk groups based on prediction
    high_risk_indices = np.where(y_pred == 1)[0]
    low_risk_indices = np.where(y_pred == 0)[0]
    
    high_risk_outcomes = y_true[high_risk_indices]
    low_risk_outcomes = y_true[low_risk_indices]
    
    # Compute Absolute Risk in each group
    abs_risk_high = np.mean(high_risk_outcomes)
    abs_risk_low = np.mean(low_risk_outcomes)
    
    # Compute Relative Risk
    rel_risk = abs_risk_high / abs_risk_low
    
    # Compute Attributable Risk
    attrib_risk = abs_risk_high - abs_risk_low
    
    # Bootstrapping for CI of Relative Risk, Absolute Risk, and Attributable Risk
    mean_rr, ci_rr_lower, ci_rr_upper = bootstrap_confidence_interval(y_true, y_pred, lambda y, y_pred: np.mean(y[y_pred==1]) / np.mean(y[y_pred==0]), n_bootstraps)
    mean_ar_high, ci_ar_high_lower, ci_ar_high_upper = bootstrap_confidence_interval(y_true, y_pred, lambda y, y_pred: np.mean(y[y_pred==1]), n_bootstraps)
    mean_ar_low, ci_ar_low_lower, ci_ar_low_upper = bootstrap_confidence_interval(y_true, y_pred, lambda y, y_pred: np.mean(y[y_pred==0]), n_bootstraps)
    mean_attrib_risk, ci_attrib_risk_lower, ci_attrib_risk_upper = bootstrap_confidence_interval(y_true, y_pred, lambda y, y_pred: np.mean(y[y_pred==1]) - np.mean(y[y_pred==0]), n_bootstraps)
    
    # Chi-squared test for comparison
    contingency_table = [[np.sum(high_risk_outcomes==1), np.sum(high_risk_outcomes==0)],
                         [np.sum(low_risk_outcomes==1), np.sum(low_risk_outcomes==0)]]
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
    
    # Print results
    print(f"Relative Risk: {rel_risk:.2f} (95% CI: {ci_rr_lower:.2f}-{ci_rr_upper:.2f})")
    print(f"Absolute Risk High: {abs_risk_high:.4f} (95% CI: {ci_ar_high_lower:.4f}-{ci_ar_high_upper:.4f})")
    print(f"Absolute Risk Low: {abs_risk_low:.4f} (95% CI: {ci_ar_low_lower:.4f}-{ci_ar_low_upper:.4f})")
    print(f"Attributable Risk: {attrib_risk:.4f} (95% CI: {ci_attrib_risk_lower:.4f}-{ci_attrib_risk_upper:.4f})")
    print(f"P-value for group comparison: {p_value:.9f}")

def plot_calibration_curve(model, model_name, X, y, n_bins=10):
    """
    Plot the calibration curve for a given model.
    
    Parameters:
    - model: The model to plot the calibration curve for.
    - model_name: Name of the model (for labeling purposes).
    - X: Test features.
    - y: True labels.
    - n_bins: Number of bins to use for calibration curve.
    """
    # Predict probabilities
    prob_pos = model.predict_proba(X)[:, 1]
    # Calculate the calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(y, prob_pos, n_bins=n_bins)

    # Plotting
    if model_name == 'Stacked Model':
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", color='black', label=f"{model_name}")
    else:
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"{model_name}")
    plt.ylabel("Fraction of positives")
    plt.xlabel("Mean predicted value")
    plt.title(f"Calibration Curve ({model_name})")
    plt.legend(loc="best")
    plt.grid(True)

def display_results(model, all_models, X, y, class_labels, target_fpr=0.5, plot_ci=True, n_bootstraps = 100):    
    # Initialize lists to store results
    model_names = []
    auc_scores = []
    auc_confidence_intervals = []
    
    plt.figure(figsize=(10, 8))
    
    # Plot ROC-AUC curves for each model
    for name, mdl in all_models.items():
        y_pred_proba = mdl.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        model_names.append(name)
        auc_scores.append(roc_auc)
            
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
        
        _ , ci_lower, ci_upper = bootstrap_confidence_interval(y, y_pred_proba, roc_auc_score, n_bootstraps)
        auc_confidence_intervals.append((ci_lower, ci_upper))
        if plot_ci:
            plt.fill_between(fpr, tpr-ci_lower, tpr+ci_upper, alpha=0.2)
            
    # Add Stacked Model separately
    y_pred_proba_stacked = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_pred_proba_stacked)
    roc_auc = auc(fpr, tpr)
    model_names.append('Stacked Model')
    auc_scores.append(roc_auc)
    
    _ , ci_lower, ci_upper = bootstrap_confidence_interval(y, y_pred_proba_stacked, roc_auc_score, n_bootstraps)
    auc_confidence_intervals.append((ci_lower, ci_upper))
    if plot_ci:
        plt.fill_between(fpr, tpr-ci_lower, tpr+ci_upper, alpha=0.2)
    
    plt.plot(fpr, tpr, label=f'Stacked Model (AUC = {roc_auc:.2f})', color='black', linewidth=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-AUC Curve')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.ylim(0, 1)
    plt.show()
    
    # Bar plot of AUC scores
    plt.figure(figsize=(10, 6))
    x_pos = np.arange(len(model_names))
    ci_lower = [max(0, auc - ci[0]) for auc, ci in zip(auc_scores, auc_confidence_intervals)]
    ci_upper = [max(0, ci[1] - auc) for auc, ci in zip(auc_scores, auc_confidence_intervals)]
    plt.bar(x_pos, auc_scores, yerr=[ci_lower, ci_upper], capsize=5)
    plt.xticks(x_pos, model_names, rotation=45, ha='right')
    plt.ylabel('AUC Score')
    plt.title('AUC Scores with 95% Confidence Intervals')
    plt.axhline(y=0.5, color='red', linestyle='--')
    plt.tight_layout()
    plt.show()
    
    # Plot calibration curves for each model
    plt.figure(figsize=(10, 8))
    plot_calibration_curve(model, 'Stacked Model', X, y, n_bins=10)
    for name, mdl in all_models.items():
        plot_calibration_curve(mdl, name, X, y, n_bins=10)
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.tight_layout()
    plt.show()
    
    # Calculate and print binary classification metrics
    for name, mdl in all_models.items():
        y_pred_proba = mdl.predict_proba(X)[:, 1]
        fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
        closest_idx = (np.abs(fpr - target_fpr)).argmin()
        threshold = thresholds[closest_idx]
        print_metrics(y, y_pred_proba, threshold, name, class_labels, n_bootstraps)
    
    # For Stacked Model
    y_pred_proba_stacked = model.predict_proba(X)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, y_pred_proba_stacked)
    closest_idx = (np.abs(fpr - target_fpr)).argmin()
    threshold = thresholds[closest_idx]
    print_metrics(y, y_pred_proba_stacked, threshold, 'Stacked Model', class_labels, n_bootstraps)

def subgroup_analysis(subgroup_column, variable_type, trained_stacked_model, trained_all_models, X_test, y_test):
    if not isinstance(X_test, pd.DataFrame):
        raise ValueError("X_test must be a pandas DataFrame.")

    # Initializing variables to store results
    subgroups = []
    model_names = ['Stacked Model'] + list(trained_all_models.keys())
    auc_scores_dict = {model_name: [] for model_name in model_names}
    
    def compute_auc_scores(subgroup_indices):
        for model_name in auc_scores_dict.keys():
            if model_name == 'Stacked Model':
                model = trained_stacked_model
            else:
                model = trained_all_models[model_name]
            auc_score = roc_auc_score(y_test[subgroup_indices], model.predict_proba(X_test.iloc[subgroup_indices])[:, 1])
            auc_scores_dict[model_name].append(auc_score)
    
    if variable_type == 'continuous':
        split_value = 0  # Assuming normalization to mean=0
        
        # Splitting the data into subgroups
        index_below = np.where(X_test[subgroup_column] <= split_value)[0]
        index_above = np.where(X_test[subgroup_column] > split_value)[0]

        # Compute AUC for each subgroup
        compute_auc_scores(index_below)
        compute_auc_scores(index_above)
        
        subgroups = ['Below or Equal Mean', 'Above Mean']

    elif variable_type == 'categorical':
        categories = [col.split('_')[-1] for col in X_test.columns if subgroup_column in col]
        others_auc_scores = {model_name: [] for model_name in model_names}
        
        for category in categories:
            column_name = f"{subgroup_column}_{category}"
            if column_name in X_test.columns:
                index_cat = np.where(X_test[column_name] == 1)[0]

                # Compute AUC for the subgroup or add to "Other"
                if len(index_cat) / len(X_test) < 0.01:
                    for model_name in others_auc_scores.keys():
                        others_auc_scores[model_name].append(roc_auc_score(y_test[index_cat], trained_all_models[model_name].predict_proba(X_test.iloc[index_cat])[:, 1] if model_name != 'Stacked Model' else trained_stacked_model.predict_proba(X_test.iloc[index_cat])[:, 1]))
                else:
                    compute_auc_scores(index_cat)
                    subgroups.append(category)

        # Handle "Other" category
        if any(others_auc_scores[model_name] for model_name in model_names):
            subgroups.append("Other")
            for model_name, scores in others_auc_scores.items():
                auc_scores_dict[model_name].append(np.mean(scores))

    else:
        raise ValueError("variable_type must be either 'continuous' or 'categorical'.")

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.1  # Width of each bar
    x = np.arange(len(subgroups))
    
    for i, model_name in enumerate(model_names):
        if model_name == 'Stacked Model':
            ax.bar(x - 0.2 + i*width, auc_scores_dict[model_name], width, color='black', label=model_name)
        else:
            ax.bar(x - 0.2 + i*width, auc_scores_dict[model_name], width, label=model_name)
    
    ax.set_xlabel('Subgroup')
    ax.set_ylabel('AUC')
    ax.set_title(f'AUC by Subgroup for {subgroup_column}')
    ax.set_xticks(x)
    ax.set_xticklabels(subgroups, rotation=45)
#     ax.legend()
    ax.legend(loc='lower center', fontsize='small')

    
    plt.tight_layout()
    plt.ylim([0,1.0])
    plt.show()

def model_interpretability(model, X_test, y_test, save_filename):
    explainer = ''
    shap_values = ''
    try:
        with open(save_filename, 'rb') as file:
            combined_data = pickle.load(file)
        explainer = combined_data["explainer"]
        shap_values = combined_data["shap_values"]
        print(f"Loaded saved shap variables for stacked model from {save_filename}")
    except:
        # Define a wrapper for the model's predict function
        def model_predict(data):
            return model.predict_proba(data)

        # Initialize SHAP Explainer with the wrapper function
        explainer = shap.Explainer(model_predict, X_test)

        # Generate SHAP values
        shap_values = explainer.shap_values(X_test)

        # Save explainer and shap_values to a file
        with open(save_filename, 'wb') as file:
            pickle.dump({"explainer": explainer, "shap_values": shap_values}, file)
    
    # SHAP Summary Plot
    try:
        # Binary classification with interest in the positive class (1)
        shap.summary_plot(shap_values[:,:,1], X_test, plot_type="bar", plot_size=(10,10))  
    except Exception as e:
        print(f"Error generating SHAP summary plot: {e}")

    return explainer, shap_values