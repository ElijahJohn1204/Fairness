import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

def accuracy(icu_data, demographic_group):
    # Calculate accuracy for each demographic group
    accuracy = icu_data.groupby(demographic_group).apply(
        lambda x: accuracy_score(x[predicted_outcome] == x[actual_outcome])
    )
    
    # Calculate disparity
    disparity = accuracy.max() - accuracy.min()

    # Check if model is fair
    is_fair = disparity < fairness_threshold

    # Plot accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    accuracy.plot(kind='bar', ax=ax)
    ax.set_title('Accuracy by Demographic Group')
    ax.set_xlabel('Demographic Group')
    ax.set_ylabel('Accuracy')

    return {
        'accuracy': accuracy,
        'disparity': disparity,
        'is_fair': is_fair,
        'fig': fig
    }

def f1(icu_data, demographic_group):
    # Calculate F1 score for each demographic group
    f1_scores = icu_data.groupby(demographic_group).apply(
        lambda x: f1_score(x[actual_outcome], x[predicted_outcome])
    )
    
    # Calculate disparity
    disparity = f1_scores.max() - f1_scores.min()

    # Check if model is fair
    is_fair = disparity < fairness_threshold

    # Plot F1 scores
    fig, ax = plt.subplots(figsize=(10, 6))
    f1_scores.plot(kind='bar', ax=ax)
    ax.set_title('F1 Score by Demographic Group')
    ax.set_xlabel('Demographic Group')
    ax.set_ylabel('F1 Score')

    return {
        'f1_scores': f1_scores,
        'disparity': disparity,
        'is_fair': is_fair,
        'fig': fig
    }

def equalized_odds(icu_data, demographic_group):
    # Calculate TPR for each demographic group
    tpr = icu_data.groupby(demographic_group)[[predicted_outcome, actual_outcome]].apply(
        lambda x: (x[predicted_outcome] & x[actual_outcome]).sum() / x[actual_outcome].sum() if x[actual_outcome].sum() > 0 else 0
    )
    
    # Calculate FPR for each demographic group
    fpr = icu_data.groupby(demographic_group)[[predicted_outcome, actual_outcome]].apply(
        lambda x: (x[predicted_outcome] & (x[actual_outcome] == 0)).sum() / ((x[actual_outcome] == 0)).sum() if (x[actual_outcome]  == 0).sum() > 0 else 0
    )
    
    # Calculate disparities
    tpr_disparity = tpr.max() - tpr.min()
    fpr_disparity = fpr.max() - fpr.min()

    # Check if model is fair
    is_fair = tpr_disparity < fairness_threshold and fpr_disparity < fairness_threshold

    # Plot TPR and FPR
    fig_tpr_fpr = plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.bar(tpr.index, tpr.values)
    plt.title('True Positive Rate (TPR)')
    plt.xlabel('Demographic Group')
    plt.ylabel('TPR')

    plt.subplot(1, 2, 2)
    plt.bar(fpr.index, fpr.values)
    plt.title('False Positive Rate (FPR)')
    plt.xlabel('Demographic Group')
    plt.ylabel('FPR')

    plt.tight_layout()

    # Plot disparities
    fig_disparities =plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.bar(['TPR Disparity'], [tpr_disparity])
    plt.title('TPR Disparity')
    plt.xlabel('Disparity')
    plt.ylabel('Value')

    plt.subplot(1, 2, 2)
    plt.bar(['FPR Disparity'], [fpr_disparity])
    plt.title('FPR Disparity')
    plt.xlabel('Disparity')
    plt.ylabel('Value')

    plt.tight_layout()

    # Return results as a dictionary
    return {
        'tpr': tpr,
        'fpr': fpr,
        'tpr_disparity': tpr_disparity,
        'fpr_disparity': fpr_disparity,
        'is_fair': is_fair,
        'fig_tpr_fpr': fig_tpr_fpr,
        'fig_disparities': fig_disparities
    }

def disparate_impact_ratio(icu_data, demographic_group):
# Calculate probability of positive outcome for each demographic group
    probabilities = df.groupby(demographic_group)[predicted_outcome].mean()

    # Calculate disparate impact ratio
    disparate_impact_ratio = probabilities.max() / probabilities.min()

    # Check if model is fair
    is_fair = disparate_impact_ratio < fairness_threshold

    # Plot positive prediction rates
    fig, ax = plt.subplots(figsize=(10, 6))
    probabilities.plot(kind='bar', ax=ax)
    ax.set_title('Positive Prediction Rate by Demographic Group')
    ax.set_xlabel('Demographic Group')
    ax.set_ylabel('Positive Prediction Rate')

    return {
        'positive_prediction_rate': probabilities,
        'disparity': disparate_impact_ratio,
        'is_fair': is_fair,
        'fig': fig
    }


def fairness_adjustment(model, X, y, fairness_metric='accuracy'):
    # Evaluate model's fairness
    fairness_score = evaluate_fairness(df, demographic)

    # Check if model is fair
    if fairness_score[fairness_metric]['is_fair'] == False:
        # Adjust model to make it fairer
        adjusted_model = adjust_model(model, X, y, fairness_metric)
        return adjusted_model
    else:
        return model

def adjust_model(model, X, y, fairness_metric):
    # Debias model using standard scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)

    # Evaluate model's fairness after debiasing
    fairness_score = evaluate_fairness(df, demographic)

    # If model is still not fair, try regularization
    if fairness_score[fairness_metric]['is_fair'] == False:
        model = LogisticRegression(penalty='l1', C=0.1)
        model.fit(X_scaled, y)

    return model

def evaluate_fairness(df, demographic):
    # Calculate fairness metrics
    accuracy = accuracy(df, demographic)
    f1_score = f1(df, demographic)
    equalized_odds = equalized_odds(df, demographic)
    disparate_impact_ratio = disparate_impact_ratio(df, demographic)

    # Return fairness metrics as a dictionary
    return {
        'accuracy': accuracy,
        'f1_score': f1_score,
        'equalized_odds': equalized_odds,
        'disparate_impact_ratio': disparate_impact_ratio
    }

np.random.seed(42)
df = pd.DataFrame({
    'predicted_mortality': np.random.randint(0, 2, size=1000),
    'actual_mortality': np.random.randint(0, 2, size=1000),
    'ethnicity': np.random.choice(['White', 'Black', 'Hispanic', 'Asian'], size=1000)
})

# Add some bias to the predicted mortality column
df['predicted_mortality'] = df['predicted_mortality'].astype(float)
df.loc[df['ethnicity'] == 'Black', 'predicted_mortality'] = np.random.randint(0, 2, size=len(df[df['ethnicity'] == 'Black'])) + 0.2

# Convert the predicted mortality column to binary (0/1)
df['predicted_mortality'] = (df['predicted_mortality'] > 0.5).astype(int)
df['actual_mortality'] = (df['actual_mortality'] > .75).astype(int)


fairness_threshold = 0.05  # Set your desired fairness threshold

# example usage:
predicted_outcome = 'predicted_mortality'  # replace with the actual column name of the predicted outcome
actual_outcome = 'actual_mortality'  # replace with the actual column name of the actual outcome
demographic = 'ethnicity'  # replace with the actual column name of the demographic group

dict = equalized_odds(df, demographic)

if dict['is_fair']:
    print("The model is fair with respect to equalized odds.")
else:
    print("The model is not fair with respect to equalized odds.")

print("TPR Disparity:", dict['tpr_disparity'])
print("FPR Disparity:", dict['fpr_disparity'])\

#(dict['fig_tpr_fpr']).show()
#(dict['fig_disparities']).show()



'''# assume 'predicted_outcome' is the predicted outcome from the mortality model
# and 'demographic_group' is the protected attribute
def demographic_parity_disparity(data, predicted_outcome_column, protected_attribute_column):
    # Calculate proportion of positive predicted outcomes for each demographic group
    proportions = data.groupby(protected_attribute_column)[predicted_outcome_column].mean()

    # Calculate difference in proportions between groups
    disparity = proportions.max() - proportions.min()

    return disparity
'''