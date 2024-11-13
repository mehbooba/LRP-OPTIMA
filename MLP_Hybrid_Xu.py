import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import lime
import lime.lime_tabular
import shap
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression

import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow warnings and info logs

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses all TensorFlow logging except errors

tf.debugging.set_log_device_placement(False)

# Load and preprocess dataset
dataset = pd.read_csv('xuetang.csv')
dataset = dataset.iloc[0:5000, :]  # Limiting dataset for quicker testing
dataset=dataset.sort_values(by=["learner_id"], ascending=True)
#data['learner_id'] = data['learner_id'].astype(int)
#TO FILTER OUT NOISY COURSES
course_freq = dataset['course_id'].value_counts()
frequency_threshold = 10
frequent_courses = course_freq[course_freq >= frequency_threshold].index.tolist()
dataset = dataset[dataset['course_id'].isin(frequent_courses)]


print(f"Total number of rows: {dataset.shape[0]}")



X = dataset[['learner_id', 'course_id', 'viewed','explored','certified','grade','nevents','ndays_act']].values
y = dataset['learner_rating'].values.reshape(-1, 1)



'''
# Normalize input featuresa
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std
'''
feature_names = ['learner_id', 'course_id', 'viewed','explored','certified','grade','nevents','ndays_act']
continuous_features = ['learner_id', 'course_id', 'viewed','explored','certified','grade','nevents','ndays_act']
categorical_features = []
categorical_indices = [feature_names.index(name) for name in []]
# Normalize only continuous features
X_continuous = dataset[continuous_features].values
X_cont_mean = X_continuous.mean(axis=0)
X_cont_std = X_continuous.std(axis=0)
X_cont_normalized = (X_continuous - X_cont_mean) / X_cont_std

# Combine normalized continuous features with categorical features
X = np.hstack((X_cont_normalized, dataset[categorical_features].values))
print("X_train.shape",X.shape)
print("y.shape",y.shape)

# Split dataset into train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define the MLP model architecture
model = Sequential([
    Dense(16, activation='relu', input_shape=(8,)),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
])

# Compile the model
model.compile(optimizer='adam', loss='mse')  # Mean Squared Error loss for regression task

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)

# LIME and SHAP explanations
shap_explainer = shap.Explainer(model, masker=shap.maskers.Independent(X_train))
lime_explainer = lime.lime_tabular.LimeTabularExplainer(X_train, mode="regression",
                                      feature_names=feature_names, 
                                      categorical_features=categorical_indices,
                                      discretize_continuous=True)




'''
num_samples = 100
predictions = model.predict(X_test[:num_samples])
actual_values = y_test[:num_samples]

# Combine predictions and actual values into a DataFrame
results_df = pd.DataFrame({
    'Actual Values': actual_values.flatten(),
    'Predictions': predictions.flatten()
})

# Write to a CSV file
results_df.to_csv('predictions_vs_actuals.csv', index=False)

print(f"Wrote {num_samples} predictions and actual values to predictions_vs_actuals.csv")
'''




# Manual LRP Implementation

def lrp_normal(model, X):
    relevance = np.zeros_like(X)  # Initialize relevance scores

    # Forward pass to get the output
    with tf.GradientTape() as tape:
        tape.watch(X)
        predictions = model(X)

    # Backward pass: Propagate the relevance score
    relevance = tape.gradient(predictions, X)

    # Normalize relevance scores to have the same sum as the predictions
    relevance = relevance / tf.reduce_sum(relevance) * tf.reduce_sum(predictions)
    return relevance.numpy()

#LRP implementation with relevances backpropagated only for proper predictions, ie, predictions with error threshold less than 10%
X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
y_tensor = tf.convert_to_tensor(y_train, dtype=tf.float32)
lrp_scores_normal = lrp_normal(model, X_train_tensor)


def lrp_enhanced(model, X, y, threshold=0.1):
    relevance = np.zeros_like(X)  # Initialize relevance scores

    # Forward pass to get the output
    with tf.GradientTape() as tape:
        tape.watch(X)
        predictions = model(X)

    # Calculate the error between actual and predicted values
    error = tf.abs(predictions - y) / tf.abs(y)
    
    # Mask to select samples with error less than the threshold
    mask = error <= threshold

    # Backward pass: Propagate the relevance score only for selected samples
    relevance = tape.gradient(predictions, X) * tf.cast(mask, tf.float32)
    
    # Normalize relevance scores to have the same sum as the predictions
    relevance = relevance / tf.reduce_sum(relevance) * tf.reduce_sum(predictions)

    return relevance.numpy()


# Compute relevance scores for the training data

X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
y_tensor = tf.convert_to_tensor(y_train, dtype=tf.float32)
lrp_scores = lrp_enhanced(model, X_train_tensor,y_tensor)

# Compute SHAP values
shap_values = shap_explainer(X_train)
norm_shap_scores = np.mean(np.abs(shap_values.values), axis=0)

# Compute LIME scores
lime_scores = np.zeros(X_train.shape[1])
for i in range(X_train.shape[0]):
    instance = X_train[i]
    
    exp = lime_explainer.explain_instance(instance, lambda x: model.predict(x, verbose=False), num_features=X_train.shape[1])

    lime_feature_scores = exp.as_map()[1]
    
    for feature_idx, score in lime_feature_scores:
        lime_scores[feature_idx] += np.abs(score)
lime_scores /= X_train.shape[0]  # Average Lime scores over all samples

# Normalize all scores
norm_lrp_scores_normal= np.mean(np.abs(lrp_scores_normal), axis=0)
norm_lrp_scores = np.mean(np.abs(lrp_scores), axis=0)
norm_lime_scores = lime_scores / np.max(lime_scores) if np.max(lime_scores) > 0 else lime_scores
norm_shap_scores = norm_shap_scores / np.max(norm_shap_scores) if np.max(norm_shap_scores) > 0 else norm_shap_scores

# Combine scores
#combined_scores = norm_lrp_scores + norm_lime_scores + norm_shap_scores
#print("Combined Scores:", combined_scores)

# FINDING PROPER COEFFICIENTS OF WEIGHTED SUM

# Generate a grid of weights
'''
weights_range = {
    'LRP': np.linspace(0, 1, 10),  # 10 values from 0 to 1
    'SHAP': np.linspace(0, 1, 10),  # 10 values from 0 to 1
    'LIME': np.linspace(0, 1, 10)   # 10 values from 0 to 1
}

# Function to compute combined scores
def compute_combined_scores(lrp_scores, shap_scores, lime_scores, w_lrp, w_shap, w_lime):
    return w_lrp * lrp_scores + w_shap * shap_scores + w_lime * lime_scores

# Objective function to minimize MSE
def objective_function(weights):
    w_lrp, w_shap, w_lime = weights
    combined_scores = compute_combined_scores(norm_lrp_scores, norm_shap_scores, norm_lime_scores, w_lrp, w_shap, w_lime)
    return -np.mean(combined_scores)  # Optimization problem is to maximize the mean combined score

# Grid search to find the best weights
best_score = 0
best_weights = [0.4,0.3,0.3]

print("WEIGHING STARTS")
for w_lrp in weights_range['LRP']:
    for w_shap in weights_range['SHAP']:
        for w_lime in weights_range['LIME']:
            weights = (w_lrp, w_shap, w_lime)
            score = objective_function(weights)
            if score > best_score:
                best_score = score
                best_weights = weights

print(f"Best Weights: LRP={best_weights[0]}, SHAP={best_weights[1]}, LIME={best_weights[2]}")
print(f"Best Score: {best_score}")
'''

w_lrp=0.4
w_shap=0.25
w_lime=0.35
combined_scores=w_lrp * norm_lrp_scores + w_shap * norm_shap_scores + w_lime * norm_lime_scores
'''
# Visualization
features = ['learner_id', 'course_id', 'n_course_avg_rating', 'n_Counts', 'n_instructr_perf', 'sentiment_score','instructional_level_encoded', 'language_encoded', '2nd_level_category_encoded']
plt.figure(figsize=(10, 9))
plt.bar(features, combined_scores)
plt.xlabel('Features')
plt.ylabel('Combined Importance Score')
plt.title('Combined Feature Importance for Learner Rating Prediction')
plt.xticks(rotation=45)
plt.show()
'''
print("Starting Model performance analysis using top features")
def evaluate_performance(features, X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train[:, features], y_train)
    predictions = model.predict(X_test[:, features])
    return mean_squared_error(y_test, predictions)
    
    
top_features_lrp_normal = np.argsort(-norm_lrp_scores_normal)[:7]  # Top 5 features from LRP
top_features_lrp_enhanced = np.argsort(-norm_lrp_scores)[:7]  # Top 5 features from LRP
top_features_shap = np.argsort(-norm_shap_scores)[:7]  # Top 5 features from SHAP
top_features_lime = np.argsort(-norm_lime_scores)[:7]  # Top 5 features from LIME
top_features_combined = np.argsort(-combined_scores)[:7]  # Top 5 features from Combined
print("top_features_lrp_normal",top_features_lrp_normal)
print("top_features_shap",top_features_shap)
print("top_features_lime",top_features_lime)
print("top_features_lrp_enhanced",top_features_lrp_enhanced)

all_features = np.argsort(-norm_lrp_scores_normal)
mse=evaluate_performance(all_features, X_train, X_test, y_train, y_test)
mse_lrp_enhanced = evaluate_performance(top_features_lrp_enhanced, X_train, X_test, y_train, y_test)
mse_lrp_normal = evaluate_performance(top_features_lrp_normal, X_train, X_test, y_train, y_test)
mse_shap = evaluate_performance(top_features_shap, X_train, X_test, y_train, y_test)
mse_lime = evaluate_performance(top_features_lime, X_train, X_test, y_train, y_test)
mse_combined = evaluate_performance(top_features_combined, X_train, X_test, y_train, y_test)
print(f"Model performance (MSE) with top features - baseline: {mse},LRP_normal: {mse_lrp_normal},LRP_enhanced: {mse_lrp_enhanced} ,SHAP: {mse_shap}, LIME: {mse_lime}, COMBINED: {mse_combined}")

'''

print("Starting Stability analysis")
def calculate_stability(method, X, y, n_iterations=100):
    importances = []
    for _ in range(n_iterations):
        X_resampled, y_resampled = resample(X, y)
        method_importances = method(X_resampled, y_resampled)  # Calculate importances using the method
        importances.append(method_importances)
    importances = np.array(importances)
    mean_importances = np.mean(importances, axis=0)
    std_importances = np.std(importances, axis=0)
    return mean_importances, std_importances

# Methods for stability calculation
def shap_method(X, y):
    shap_values = shap_explainer(X)
    return np.mean(np.abs(shap_values.values), axis=0)

def lime_method(X, y):
    lime_importances = np.zeros(X.shape[1])
    for i in range(X.shape[0]):
        exp = lime_explainer.explain_instance(X[i], lambda x: model.predict(x, verbose=False), num_features=X.shape[1])
        feature_scores = exp.as_map()[1]
        for feature_idx, score in feature_scores:
            lime_importances[feature_idx] += np.abs(score)
    lime_importances /= X.shape[0]
    return lime_importances

def lrp_method(X, y):
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    lrp_scores = lrp(model, X_tensor)
    return np.mean(np.abs(lrp_scores), axis=0)

def combined_method(X, y, weights):
    w_lrp, w_shap, w_lime = weights
    lrp_scores = lrp_method(X, y)
    shap_scores = shap_method(X, y)
    lime_scores = lime_method(X, y)
    combined_scores = compute_combined_scores(lrp_scores, shap_scores, lime_scores, w_lrp, w_shap, w_lime)
    return combined_scores

def combined_stability(X, y, weights, n_iterations=100):
    combined_importances = []
    for _ in range(n_iterations):
        X_resampled, y_resampled = resample(X, y)
        combined_importances.append(combined_method(X_resampled, y_resampled, weights))
    combined_importances = np.array(combined_importances)
    mean_combined_importances = np.mean(combined_importances, axis=0)
    std_combined_importances = np.std(combined_importances, axis=0)
    return mean_combined_importances, std_combined_importances

# Calculate stability for different methods
mean_importances_lrp, std_importances_lrp = calculate_stability(lrp_method, X_train, y_train)
mean_importances_shap, std_importances_shap = calculate_stability(shap_method, X_train, y_train)
mean_importances_lime, std_importances_lime = calculate_stability(lime_method, X_train, y_train)

# Calculate stability for combined method
mean_importances_combined, std_importances_combined = combined_stability(X_train, y_train, best_weights)

# Print stability results
print("Stability (Mean ± Std) - LRP: ", mean_importances_lrp, std_importances_lrp)
print("Stability (Mean ± Std) - SHAP: ", mean_importances_shap, std_importances_shap)
print("Stability (Mean ± Std) - LIME: ", mean_importances_lime, std_importances_lime)
print("Stability (Mean ± Std) - Combined: ", mean_importances_combined, std_importances_combined)


combined_scores_2d = combined_scores.reshape(1, -1) 
min_val = np.min(combined_scores_2d)
max_val = np.max(combined_scores_2d)

# Normalize the array
combined_scores_normalized = (combined_scores_2d - min_val) / (max_val - min_val)
# Plot heatmap
sns.heatmap(combined_scores_normalized, cmap='Blues', cbar=True, annot=True, fmt=".2f",
            xticklabels=["LID","CID","NCAR","NC","NIP","SS","IL","LA","SLC"], yticklabels=False,cbar_kws={"orientation": "horizontal"})  # yticklabels=False to remove y-axis labels

#plt.title('Feature Importance Heatmap')
plt.xlabel('Features')
plt.ylabel('Mean Absolute Gradient')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save the figure
num_rows = dataset.shape[0]  # Get the number of rows from the dataframe
filename = f"MLP-Hybrid-heatmap-{num_rows}.jpg"  # Create the filename with the number of rows
plt.savefig(filename, bbox_inches='tight', dpi=300)  # Save the plot with the generated filename
plt.show()
'''
