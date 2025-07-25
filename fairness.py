import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

def accuracy(icu_data, demographic_group, predicted_outcome, actual_outcome, fairness_threshold):
    # Calculate accuracy for each demographic group
    accuracy = icu_data.groupby(demographic_group).apply(
        lambda x: accuracy_score(x[predicted_outcome], x[actual_outcome])
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

def f1(icu_data, demographic_group, predicted_outcome, actual_outcome, fairness_threshold):
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

def equalized_odds(icu_data, demographic_group, predicted_outcome, actual_outcome, fairness_threshold):
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

    # Return results as a dictionary
    return {
        'tpr': tpr,
        'fpr': fpr,
        'tpr_disparity': tpr_disparity,
        'fpr_disparity': fpr_disparity,
        'is_fair': is_fair,
        'fig_tpr_fpr': fig_tpr_fpr
    }

def disparate_impact_ratio(icu_data, demographic_group, predicted_outcome, actual_outcome, fairness_threshold):
    # Calculate probability of positive outcome for each demographic group
    probabilities = icu_data.groupby(demographic_group)[predicted_outcome].mean()

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

'''
def fairness_adjustment(model, X, y, fairness_metric, demographic, predicted_outcome, actual_outcome, fairness_threshold):
    # Evaluate model's fairness
    fairness_score = evaluate_fairness(df, demographic)

    # Check if model is fair
    if fairness_score[fairness_metric]['is_fair'] == False:
        # Adjust model to make it fairer
        adjusted_model = adjust_model(model, X, y, fairness_metric)
        return adjusted_model
    else:
        return model

def adjust_model(model, X, y, fairness_metric, demographic, predicted_outcome, actual_outcome, fairness_threshold):
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
'''

def evaluate_fairness(df, demographic, predicted_outcome, actual_outcome, fairness_threshold, fairness_metric='all', ratio_threshold = 1.2):
    # Calculate fairness metrics
    if fairness_metric == 'all':
        return {
            'accuracy': accuracy(df, demographic, predicted_outcome, actual_outcome, fairness_threshold),
            'f1_score': f1(df, demographic, predicted_outcome, actual_outcome, fairness_threshold),
            'equalized_odds': equalized_odds(df, demographic, predicted_outcome, actual_outcome, fairness_threshold),
            'disparate_impact_ratio': disparate_impact_ratio(df, demographic, predicted_outcome, actual_outcome, ratio_threshold)
        }
    elif fairness_metric == 'accuracy':
        return accuracy(df, demographic, predicted_outcome, actual_outcome, fairness_threshold)
    elif fairness_metric == 'f1_score':
        return f1_score(df, demographic, predicted_outcome, actual_outcome, fairness_threshold)
    elif fairness_metric == 'equalized_odds':    
        return equalized_odds(df, demographic, predicted_outcome, actual_outcome, fairness_threshold)
    elif fairness_metric == 'disparate_impact_ratio':
        return disparate_impact_ratio(df, demographic, predicted_outcome, actual_outcome, ratio_threshold)


def roc_analysis(df, demographic, predicted_outcome, actual_outcome):
    demographics = df[demographic].unique()
    roc_curves = {}
    
    for demographic_group in demographics:
        df_group = df[df[demographic] == demographic_group]
        y_pred_proba = df_group[predicted_outcome]
        y_true = df_group[actual_outcome]
        
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_curves[demographic_group] = (fpr, tpr, thresholds)
    
    plot_roc_curves(roc_curves)

    return roc_curves

def plot_roc_curves(roc_curves):
    for demographic_group, (fpr, tpr, thresholds) in roc_curves.items():
        plt.plot(fpr, tpr, label=demographic_group)
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.show()

def calibration_analysis(df, demographic, predicted_outcome, actual_outcome):
    demographics = df[demographic].unique()
    calibration_plots = {}
    
    for demographic_group in demographics:
        df_group = df[df[demographic] == demographic_group]
        y_pred_proba = df_group[predicted_outcome]
        y_true = df_group[actual_outcome]
        
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_pred_proba, n_bins=10)
        calibration_plots[demographic_group] = (fraction_of_positives, mean_predicted_value)
    
    plot_calibration_plots(calibration_plots)
    return calibration_plots

def plot_calibration_plots(calibration_plots):
    for demographic_group, (fraction_of_positives, mean_predicted_value) in calibration_plots.items():
        plt.plot(mean_predicted_value, fraction_of_positives, label=demographic_group)
    
    plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
    plt.xlabel('Mean predicted value')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration plots')
    plt.legend()
    plt.show()

def decision_analysis(df, demographic, predicted_outcome, actual_outcome):
    demographics = df[demographic].unique()
    decision_curves = {}
    
    for demographic_group in demographics:
        df_group = df[df[demographic] == demographic_group]
        y_pred_proba = df_group[predicted_outcome]
        y_true = df_group[actual_outcome]
        
        thresholds = np.linspace(0, 1, 100)
        net_benefits = []
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            true_positives = np.sum(y_true * y_pred)
            false_positives = np.sum((1 - y_true) * y_pred)
            net_benefit = true_positives - false_positives
            net_benefits.append(net_benefit)
        
        decision_curves[demographic_group] = (thresholds, net_benefits)
    
    plot_decision_curves(decision_curves)
    return decision_curves

def plot_decision_curves(decision_curves):
    for demographic_group, (thresholds, net_benefits) in decision_curves.items():
        plt.plot(thresholds, net_benefits, label=demographic_group)
    
    plt.xlabel('Threshold')
    plt.ylabel('Net Benefit')
    plt.title('Decision Analysis Curves')
    plt.legend()
    plt.show()