import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
import lime
import lime.lime_tabular
import shap
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Suppress TensorFlow warnings and info logs
tf.get_logger().setLevel('ERROR')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses all TensorFlow logging except errors
tf.debugging.set_log_device_placement(False)

# Load and preprocess dataset
dataset = pd.read_csv('xuetang.csv')
dataset = dataset.iloc[0:7500, :]  # Limiting dataset for quicker testing
dataset = dataset.sort_values(by=["learner_id"], ascending=True)

# Filter out noisy courses
course_freq = dataset['course_id'].value_counts()
frequency_threshold = 10
frequent_courses = course_freq[course_freq >= frequency_threshold].index.tolist()
dataset = dataset[dataset['course_id'].isin(frequent_courses)]


print(f"Total number of rows: {dataset.shape[0]}")

# Prepare features and target
feature_names = ['learner_id', 'course_id', 'viewed','explored','certified','grade','nevents','ndays_act']
continuous_features = ['learner_id', 'course_id', 'viewed','explored','certified','grade','nevents','ndays_act']
categorical_features = []
X = dataset[feature_names].values
y = dataset['learner_rating'].values.reshape(-1, 1)


categorical_indices = [
    feature_names.index(name) for name in categorical_features
]

# Normalize only continuous features
X_continuous = dataset[continuous_features].values
scaler = StandardScaler()
X_cont_normalized = scaler.fit_transform(X_continuous)

# Combine normalized continuous features with categorical features
X = np.hstack((X_cont_normalized, dataset[categorical_features].values))

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape data for GRU [samples, time_steps, features]
X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Define the GRU model architecture
model = Sequential([
    GRU(50, activation='relu', input_shape=(1, X_train.shape[1])),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)

# Define a prediction function for SHAP
def predict_fn(x):
    x_reshaped = tf.convert_to_tensor(x, dtype=tf.float32)
    x_reshaped = tf.reshape(x_reshaped, (x_reshaped.shape[0], 1, x_reshaped.shape[1]))
    return model.predict(x_reshaped, verbose=0).flatten()




'''
num_samples = 100
X_test_temp=X_test
X_test_reshaped = X_test_temp.reshape((X_test_temp.shape[0], 1, X_test_temp.shape[1])) 
predictions = model.predict(X_test_reshaped[:num_samples])
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


# Prepare SHAP explainer using KernelExplainer
background_data = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]  # Subset of training data as background
shap_explainer = shap.KernelExplainer(predict_fn, background_data)

# Compute SHAP values
shap_values = shap_explainer.shap_values(X_train)

# SHAP Values Post-Processing
shap_values_flat = np.mean(np.abs(shap_values), axis=0)
norm_shap_scores = shap_values_flat / np.max(shap_values_flat) if np.max(shap_values_flat) > 0 else shap_values_flat

# LIME and Manual LRP Implementation
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train, mode="regression",
    feature_names=feature_names, 
    categorical_features=categorical_indices,
    discretize_continuous=True
)

def lrp_normal(model, X):
    relevance = np.zeros_like(X)  # Initialize relevance scores

    # Forward pass to get the output
    with tf.GradientTape() as tape:
        tape.watch(X)
        predictions = model(X)

    # Backward pass: Propagate the relevance score
    relevance = tape.gradient(predictions, X)

    # Normalize relevance scores to sum to 1 across features
    relevance = relevance / tf.reduce_sum(relevance, axis=1, keepdims=True)
    return relevance.numpy()

# Compute LRP relevance scores for the training data
X_train_tensor = tf.convert_to_tensor(X_train_reshaped, dtype=tf.float32)
lrp_scores_normal = lrp_normal(model, X_train_tensor)
print("lrp_scores_normal", lrp_scores_normal.shape)
lrp_scores_normal_flat = lrp_scores_normal.reshape(lrp_scores_normal.shape[0], -1)  # Flattening

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

# Compute LRP relevance scores for the training data
X_train_tensor = tf.convert_to_tensor(X_train_reshaped, dtype=tf.float32)
y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.float32)

lrp_scores_enhanced = lrp_enhanced(model, X_train_tensor, y_train_tensor)
print("lrp_scores_enhanced", lrp_scores_normal.shape)
lrp_scores_enhanced_flat = lrp_scores_normal.reshape(lrp_scores_enhanced.shape[0], -1)  # Flattening
lrp_scores_normal = np.mean(np.abs(lrp_scores_normal_flat), axis=0)

lrp_scores_enhanced = np.mean(np.abs(lrp_scores_enhanced_flat), axis=0)

# Compute LIME scores
lime_scores = np.zeros(X_train.shape[1])
for i in range(X_train.shape[0]):
    instance = X_train[i]
    exp = lime_explainer.explain_instance(instance, lambda x: model.predict(x.reshape((x.shape[0], 1, x.shape[1])), verbose=False), num_features=X_train.shape[1])
    lime_feature_scores = exp.as_map()[1]
    for feature_idx, score in lime_feature_scores:
        lime_scores[feature_idx] += np.abs(score)
lime_scores /= X_train.shape[0]  # Average Lime scores over all samples
'''
# Debugging: Print lengths and contents to ensure consistency
print("Feature names length:", len(feature_names))
print("SHAP scores length:", len(norm_shap_scores))
print("LIME scores length:", len(lime_scores))
print("LRP scores normal length:", len(np.mean(np.abs(lrp_scores_normal_flat), axis=0)))
print("LRP scores normal length:", len(np.mean(np.abs(lrp_scores_enhanced_flat), axis=0)))
'''
# Combine scores
w_lrp = 0.4
w_shap = 0.25
w_lime = 0.35
combined_scores = w_lrp * np.mean(np.abs(lrp_scores_normal), axis=0) + w_shap * norm_shap_scores + w_lime * lime_scores
combined_scores_enhanced = w_lrp * np.mean(np.abs(lrp_scores_enhanced), axis=0) + w_shap * norm_shap_scores + w_lime * lime_scores

# Ensure combined_scores is 1-dimensional
combined_scores = np.ravel(combined_scores)
combined_scores_enhanced = np.ravel(combined_scores_enhanced)
# Check lengths
assert len(feature_names) == len(combined_scores), "Feature names and scores must have the same length"

# Create a DataFrame for easier plotting with seaborn
plot_data = pd.DataFrame({
    'Feature': feature_names,
    'Importance': combined_scores
})

# Evaluate model performance
def evaluate_performance(top_features, X_train, X_test, y_train, y_test):
    model = Sequential([
        GRU(50, activation='relu', input_shape=(1, len(top_features))),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train[:, :, top_features], y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)

    y_pred = model.predict(X_test[:, :, top_features], verbose=0)
    mse = mean_squared_error(y_test, y_pred)
    return mse

top_features_lrp_normal = np.argsort(-lrp_scores_normal)[:7]  # Top 5 features from LRP
top_features_lrp_enhanced = np.argsort(-lrp_scores_enhanced)[:7]  # Top 5 features from LRP
top_features_shap = np.argsort(-norm_shap_scores)[:7]  # Top 5 features from SHAP
top_features_lime = np.argsort(-lime_scores)[:7]  # Top 5 features from LIME
top_features_combined = np.argsort(-combined_scores)[:7]  # Top 5 features from Combined
top_features_combined_enhanced = np.argsort(-combined_scores_enhanced)[:7]  # Top 5 features from Combined

print("top_features_lrp_normal",top_features_lrp_normal)
print("top_features_shap",top_features_shap)
print("top_features_lime",top_features_lime)
print("top_features_lrp_enhanced",top_features_lrp_enhanced)


all_features = np.argsort(-lrp_scores_normal)[:9]
mse=evaluate_performance(all_features,  X_train_reshaped, X_test_reshaped, y_train, y_test)
mse_lrp_enhanced = evaluate_performance(top_features_lrp_enhanced, X_train_reshaped, X_test_reshaped, y_train, y_test)
mse_lrp_normal = evaluate_performance(top_features_lrp_normal, X_train_reshaped, X_test_reshaped, y_train, y_test)
mse_shap = evaluate_performance(top_features_shap, X_train_reshaped, X_test_reshaped, y_train, y_test)
mse_lime = evaluate_performance(top_features_lime, X_train_reshaped, X_test_reshaped, y_train, y_test)
mse_combined = evaluate_performance(top_features_combined, X_train_reshaped, X_test_reshaped, y_train, y_test)
mse_combined_enhanced = evaluate_performance(top_features_combined_enhanced, X_train_reshaped, X_test_reshaped, y_train, y_test)

print(f"Model performance (MSE) with top features - baseline: {mse},LRP: {mse_lrp_normal}, LRP Enhanced: {mse_lrp_enhanced}, SHAP: {mse_shap}, LIME: {mse_lime}, COMBINED: {mse_combined}, COMBINED_ENHANCED: {mse_combined_enhanced}")
'''
combined_scores_2d = combined_scores_enhanced.reshape(1, -1)
min_val = np.min(combined_scores_2d)
max_val = np.max(combined_scores_2d)

# Normalize the array
combined_scores_normalized = (combined_scores_2d - min_val) / (max_val - min_val)
# Plot heatmap
sns.heatmap(combined_scores_normalized, cmap='Blues', cbar=True, annot=True, fmt=".2f",
            xticklabels=["LID", "CID", "NCAR", "NC", "NIP", "SS", "IL", "LA", "SLC"], yticklabels=False, cbar_kws={"orientation": "horizontal"})  # yticklabels=False to remove y-axis labels

# plt.title('Feature Importance Heatmap')
plt.xlabel('Features')
plt.ylabel('Mean Absolute Gradient')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save the figure
num_rows = dataset.shape[0]  # Get the number of rows from the dataframe
filename = f"GRU-Hybrid-heatmap-{num_rows}.jpg"  # Create the filename with the number of rows
plt.savefig(filename, bbox_inches='tight', dpi=300)  # Save the plot with the generated filename
plt.show()
'''

