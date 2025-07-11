import numpy as np
import pandas as pd
import fairness as fair

race_map = {
    'White': 'White',
    'Black or African American': 'Black',
    'Asian': 'Asian',
    'Other': 'Others',
    'Unknown': 'Others',
    'Did Not Encounter': 'Others',
    'Refusal': 'Others',
    'American Indian or Alaska Native': 'Others',
    'Native Hawaiian or Other Pacific Islander': 'Others',
    np.nan: 'Others'
}

ethnicity_map = {
    'Non-Hispanic': 'Not Hispanic or Latino',
    'Hispanic': 'Hispanic or Latino',
    'Did Not Encounter': 'Not Hispanic or Latino',
    'Refusal': 'Not Hispanic or Latino',
    'Unknown': 'Not Hispanic or Latino',
    np.nan: 'Not Hispanic or Latino'
}


location = pd.read_parquet('C:/Users/ejohn/Downloads/CLIF_MIMIC/data/clif_adt.parquet')
encounter = pd.read_parquet('C:/Users/ejohn/Downloads/CLIF_MIMIC/data/clif_hospitalization.parquet')
labs = pd.read_parquet('C:/Users/ejohn/Downloads/CLIF_MIMIC/data/clif_labs.parquet')
demog = pd.read_parquet('C:/Users/ejohn/Downloads/CLIF_MIMIC/data/clif_patient.parquet')
vitals = pd.read_parquet('C:/Users/ejohn/Downloads/CLIF_MIMIC/data/clif_vitals.parquet')


# Merge the dataframes to create a comprehensive ICU dataset
icu_data=pd.merge(location[['hospitalization_id','location_category','in_dttm','out_dttm']],\
              encounter[['hospitalization_id','patient_id','admission_dttm', 'age_at_admission', 'discharge_category']], on=['hospitalization_id'], how='left')


# Define mappings
icu_data['in_dttm'] = pd.to_datetime(icu_data['in_dttm'])
icu_data['admission_dttm'] = pd.to_datetime(icu_data['admission_dttm'])
icu_data['out_dttm'] = pd.to_datetime(icu_data['out_dttm'])
icu_data = icu_data[icu_data['age_at_admission'].notna()]

icu_48hr_check = icu_data[
    (icu_data['location_category'] == 'icu') &
    (icu_data['in_dttm'] >= icu_data['admission_dttm']) &
    (icu_data['in_dttm'] <= icu_data['admission_dttm'] + pd.Timedelta(hours=48)) & 
    (icu_data['age_at_admission'].notna()) &
    (icu_data['age_at_admission'] >= 18)
]['hospitalization_id'].unique()

icu_data=icu_data[icu_data['hospitalization_id'].isin(icu_48hr_check) & (icu_data['in_dttm'] <= icu_data['admission_dttm'] + pd.Timedelta(hours=72))].reset_index(drop=True)


icu_data = icu_data.sort_values(by=['in_dttm']).reset_index(drop=True)

icu_data["RANK"]=icu_data.sort_values(by=['in_dttm'], ascending=True).groupby("hospitalization_id")["in_dttm"].rank(method="first", ascending=True).astype(int)


min_icu=icu_data[icu_data['location_category'] == 'icu'].groupby('hospitalization_id')['RANK'].min()
icu_data=pd.merge(icu_data, pd.DataFrame(zip(min_icu.index, min_icu.values), columns=['hospitalization_id', 'min_icu']), on='hospitalization_id', how='left')
icu_data=icu_data[icu_data['RANK']>=icu_data['min_icu']].reset_index(drop=True)

icu_data.loc[icu_data['location_category'] == 'or', 'location_category'] = 'icu'

icu_data['group_id'] = (icu_data.groupby('hospitalization_id')['location_category'].shift() != icu_data['location_category']).astype(int)
icu_data['group_id'] = icu_data.sort_values(by=['in_dttm'], ascending=True).groupby('hospitalization_id')['group_id'].cumsum()

icu_data = icu_data.sort_values(by=['in_dttm'], ascending=True).groupby(['hospitalization_id', 'patient_id', 'location_category', 'group_id']).agg(
    min_in_dttm=('in_dttm', 'min'),
    max_out_dttm=('out_dttm', 'max'),
    admission_dttm=('admission_dttm', 'first'),
    age=('age_at_admission', 'first'),
    dispo=('discharge_category', 'first')
).reset_index()


min_icu=icu_data[icu_data['location_category'] == 'icu'].groupby('hospitalization_id')['group_id'].min()
icu_data=pd.merge(icu_data, pd.DataFrame(zip(min_icu.index, min_icu.values), columns=['hospitalization_id', 'min_icu']), on='hospitalization_id', how='left')

icu_data=icu_data[(icu_data['min_icu']==icu_data['group_id']) &
         (icu_data['max_out_dttm']-icu_data['min_in_dttm'] >= pd.Timedelta(hours=24))
         ].reset_index(drop=True)

icu_data['after_24hr']=icu_data['min_in_dttm'] + pd.Timedelta(hours=24)

icu_data=icu_data[['hospitalization_id', 'patient_id', 'min_in_dttm','max_out_dttm','admission_dttm','after_24hr','age','dispo']]

# Merge with demographic data
icu_data=pd.merge(icu_data,\
                  demog, on=['patient_id'], how='left')[['hospitalization_id','min_in_dttm','after_24hr','admission_dttm','max_out_dttm','age','dispo','sex_category','ethnicity_category','race_category']]
icu_data=icu_data[~icu_data['sex_category'].isna()].reset_index(drop=True)
icu_data['isfemale']=(icu_data['sex_category'].str.lower() == 'female').astype(int)
icu_data['isdeathdispo'] = (icu_data['dispo'].str.contains('dead|expired', case=False, regex=True)).astype(int)

icu_data['ethnicity_category'] = icu_data['ethnicity_category'].map(ethnicity_map)
icu_data['race_category'] = icu_data['race_category'].map(race_map)
icu_data['ICU_stay_hrs']= (icu_data['max_out_dttm'] - icu_data['min_in_dttm']).dt.total_seconds() / 3600

vitals = vitals[['hospitalization_id', 'recorded_dttm', 'vital_category', 'vital_value']]
vitals['recorded_dttm'] = pd.to_datetime(vitals['recorded_dttm'])
vitals['vital_value'] = pd.to_numeric(vitals['vital_value'])

valid_vals = ['weight_kg', 'heart_rate', 'sbp', 'dbp', 'temp_c','height_cm']
valid_ids = icu_data['hospitalization_id'].unique()

vitals = vitals[vitals['hospitalization_id'].isin(valid_ids) 
                & vitals['vital_category'].isin(valid_vals)]

vitals = (
    vitals
    .sort_values(['hospitalization_id', 'recorded_dttm'])  # ensures "first" makes sense
    .drop_duplicates(subset=['hospitalization_id', 'recorded_dttm', 'vital_category'])
    .pivot_table(
        index=['hospitalization_id', 'recorded_dttm'],
        columns='vital_category',
        values='vital_value',
        aggfunc='first'
    )
    .reset_index()  # optional, if you want hospitalization_id & datetime back as columns
)


vitals['height_meters'] = vitals['height_cm'] * .01
vitals['bmi'] = vitals['weight_kg'] / (vitals['height_meters'] ** 2)


icu_data_agg=pd.merge(icu_data,vitals, on=['hospitalization_id'], how='left')
icu_data_agg=icu_data_agg[(icu_data_agg['recorded_dttm'] >= icu_data_agg['min_in_dttm']) & (icu_data_agg['recorded_dttm'] <= icu_data_agg['after_24hr'])].reset_index(drop=True)

icu_data_agg = icu_data_agg.groupby(['hospitalization_id']).agg(
    min_bmi=('bmi', 'min'),
    max_bmi=('bmi', 'max'),
    avg_bmi=('bmi', 'mean'),
    min_weight_kg=('weight_kg', 'min'),
    max_weight_kg=('weight_kg', 'max'),
    avg_weight_kg=('weight_kg', 'mean'),
    min_pulse=('heart_rate', 'min'),
    max_pulse=('heart_rate', 'max'),
    avg_pulse=('heart_rate', 'mean'),
    min_sbp=('sbp', 'min'),
    max_sbp=('sbp', 'max'),
    avg_sbp=('sbp', 'mean'),
    min_dbp=('dbp', 'min'),
    max_dbp=('dbp', 'max'),
    avg_dbp=('dbp', 'mean'),
    min_temp_c=('temp_c', 'min'),
    max_temp_c=('temp_c', 'max'),
    avg_temp_c=('temp_c', 'mean'),
).reset_index()

icu_data=pd.merge(icu_data,icu_data_agg, on=['hospitalization_id'], how='left')


# Filter labs data
labs = labs[['hospitalization_id', 'lab_result_dttm', 'lab_category', 'reference_unit', 'lab_value']]
labs['lab_result_dttm'] = pd.to_datetime(labs['lab_result_dttm'])
labs['lab_value'] = pd.to_numeric(labs['lab_value'], errors='coerce')


# List of lab filters as (category, unit)
lab_filters = [
    ('monocytes_percent', '%'), ('lymphocytes_percent', '%'), ('basophils_percent', '%'), ('neutrophils_percent', '%'),
    ('albumin', 'g/dL'), ('ast', 'IU/L'), ('total_protein', 'g/dL'),
    ('alkaline_phosphatase', 'IU/L'), ('bilirubin_total', 'mg/dL'),
    ('bilirubin_conjugated', 'mg/dL'), ('calcium_total', 'mg/dL'), ('chloride', 'mEq/L'),
    ('potassium', 'mEq/L'), ('sodium', 'mEq/L'), ('glucose_serum', 'mg/dL'),
    ('hemoglobin', 'g/dL'), ('platelet_count', 'K/uL'), ('wbc', 'K/uL')
]

valid_ids = icu_data['hospitalization_id'].unique()


# Filter labs
labs = labs[
    (labs[['lab_category', 'reference_unit']].apply(tuple, axis=1).isin(lab_filters)) &
    (labs['hospitalization_id'].isin(valid_ids))
]

labs = labs[['hospitalization_id', 'lab_result_dttm', 'lab_value', 'lab_category']]

labs['lab_value'] = labs['lab_value'].fillna(0)  # Or another appropriate value
labs = (
    labs
    .sort_values(['hospitalization_id', 'lab_result_dttm'])  # ensures "first" makes sense
    .drop_duplicates(subset=['hospitalization_id', 'lab_result_dttm', 'lab_category'])
    .pivot_table(
        index=['hospitalization_id', 'lab_result_dttm'],
        columns='lab_category',
        values='lab_value',
        aggfunc='first'
    )
    .reset_index()  # optional, if you want hospitalization_id & datetime back as columns
)

icu_data_agg=pd.merge(icu_data,labs, on=['hospitalization_id'], how='left')
icu_data_agg=icu_data_agg[(icu_data_agg['lab_result_dttm'] >= icu_data_agg['min_in_dttm']) & (icu_data_agg['lab_result_dttm'] <= icu_data_agg['after_24hr'])].reset_index(drop=True)

Lab_variables = [
   'albumin', 'alkaline_phosphatase',
       'ast', 'basophils_percent', 'bilirubin_conjugated', 'bilirubin_total', 'calcium_total',
       'chloride', 'glucose_serum', 'hemoglobin', 'lymphocytes_percent', 'monocytes_percent',
       'neutrophils_percent', 'platelet_count', 'potassium', 'sodium', 'total_protein',
       'wbc'
]
agg_dict = {var: ['min', 'max', 'mean'] for var in Lab_variables}

icu_data_agg = icu_data_agg.groupby('hospitalization_id').agg(agg_dict).reset_index()

icu_data_agg.columns = ['_'.join(col).strip() if col[1] else col[0] for col in icu_data_agg.columns.values]
icu_data=pd.merge(icu_data,icu_data_agg, on=['hospitalization_id'], how='left')

demographics = icu_data[['hospitalization_id', 'isfemale', 'ethnicity_category', 'race_category']]
dem_csv = demographics.to_csv('C:/Users/ejohn/Downloads/CLIF_MIMIC/data/demographics.csv', index=False)

