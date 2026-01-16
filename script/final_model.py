# # 🍷 Wine Quality Analysis
# ### Domain: Alcoholic Beverage Sector


# ### Import Libraries


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings("ignore")

# ### Load Dataset


df = pd.read_csv("data\winequality.csv")
df.head(5)

# ### Data Cleaning


df.isnull().sum()

# ### Observation :
# The dataset contains NO null values


# ### Duplicate Records Check


df.duplicated().sum()

# ### Observation :
# The dataset contains 1,177 duplicate rows.


# ### Dataset Shape Before Removing Duplicates


df.shape

# ### Removing Duplicate Records


df = df.drop_duplicates()
df.duplicated().sum()

# ### Dataset Shape After Removing Duplicates


df.shape

# #### Observation:
# - 1,177 duplicate records were removed.
# - Final dataset contains 5,320 unique rows and 14 columns.
# - The dataset is now clean and ready for EDA and modeling.


# ### Encode Color Column


df['color'] = df['color'].map({'red': 0, 'white': 1})

# ### Data Distribution Analysis


num_cols = df.select_dtypes(include=['int64', 'float64']).columns

plt.figure(figsize=(16, 12))
for i, col in enumerate(num_cols, 1):
    plt.subplot(4, 4, i)
    sns.histplot(df[col], kde=True)
    plt.title(col)

plt.tight_layout()
plt.show()

# ### Observation:
# - Several features such as residual sugar, chlorides, and sulfur dioxide are right-skewed.
# - Features like pH and density show relatively normal distributions.


# ### Correlation analysis with wine quality
# ##### Identify features with strong positive or negative impact


# Correlation with quality
corr_with_quality = df.corr()['quality'].sort_values(ascending=False)
print(corr_with_quality)

#  ### Feature Relationship with Wine Quality


# Heatmap for correlation with quality
plt.figure(figsize=(8, 6))
sns.heatmap(
    df[num_cols].corr()[['quality']],
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation of Features with Wine Quality")
plt.show()

# ### Observation:
# - Positive correlation indicates features that improve wine quality.
# - Negative correlation indicates features that reduce wine quality.


# ### Create Quality Categories
#  MODIFIED: 3 balanced categories
# - ≤ 5 → Average
# - 6-7 → Good
# - ≥ 8 → Excellent


def quality_category(score):
    if score <= 5:
        return "Average"
    elif score <= 7:
        return "Good"
    else:
        return "Excellent"

df['quality_category'] = df['quality'].apply(quality_category)

# ### Check Distribution


print("\nQuality Category Distribution:")
print(df['quality_category'].value_counts())
print("\nQuality Score Distribution:")
print(df['quality'].value_counts().sort_index())

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.countplot(x='quality_category', data=df, order=['Average', 'Good', 'Excellent'])
plt.title("Wine Quality Category Distribution")
plt.xticks(rotation=30)

plt.subplot(1, 2, 2)
sns.countplot(x='quality', data=df)
plt.title("Wine Quality Score Distribution")

plt.tight_layout()
plt.show()

# ### Key Insights from EDA
# 
# - Alcohol shows a strong positive correlation with wine quality.
# - Volatile acidity has a strong negative impact on wine quality.
# - Most wines fall into the average quality range (scores 5–6), indicating class imbalance.
# 
# ### Business Insights & Recommendations
# 
# - Focus on controlling alcohol content, as higher alcohol is associated with better quality wines.
# - Reduce volatile acidity during fermentation to avoid poor taste and quality degradation.
# - Monitor sulphates and citric acid levels to enhance flavor balance and preservation.
# 


# ### Save Clean Dataset


df.to_csv("wine_quality_categorized.csv", index=False)

# ---


# # Model Building


# ### Feature & Target Selection


X = df.drop(columns=['quality', 'quality_category','good'])
y = df['quality_category']

# ### Train–Test Split


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTraining set distribution:")
print(y_train.value_counts())
print("\nTest set distribution:")
print(y_test.value_counts())

# ### Feature Scaling


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ### Handle Class Imbalance with RandomOverSampler
#  Using RandomOverSampler instead of SMOTE to avoid k_neighbors issue


from imblearn.over_sampling import RandomOverSampler

print("\nApplying Random Over-Sampling to handle class imbalance...")
ros = RandomOverSampler(random_state=42)
X_train_balanced, y_train_balanced = ros.fit_resample(X_train, y_train)

print("\nAfter Over-Sampling:")
print(pd.Series(y_train_balanced).value_counts())

# Scale the balanced data
X_train_balanced_scaled = scaler.fit_transform(X_train_balanced)

# ## Train Models


from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("\nTraining models with balanced data...")

#  #### KNN


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_balanced_scaled, y_train_balanced)
knn_pred = knn.predict(X_test_scaled)

#  #### Logistic Regression


lr = LogisticRegression(max_iter=1000, class_weight='balanced')
lr.fit(X_train_balanced_scaled, y_train_balanced)
lr_pred = lr.predict(X_test_scaled)

# #### SVM


svm = SVC(kernel='rbf', class_weight='balanced')
svm.fit(X_train_balanced_scaled, y_train_balanced)
svm_pred = svm.predict(X_test_scaled)

# #### Decision Tree


dt = DecisionTreeClassifier(class_weight='balanced', random_state=42)
dt.fit(X_train_balanced, y_train_balanced)
dt_pred = dt.predict(X_test)

# #### Random Forest


rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    class_weight='balanced',
    random_state=42
)

rf.fit(X_train_balanced, y_train_balanced)
rf_pred = rf.predict(X_test)


# ## Model Comparison


results = {
    "KNN": accuracy_score(y_test, knn_pred),
    "Logistic Regression": accuracy_score(y_test, lr_pred),
    "SVM": accuracy_score(y_test, svm_pred),
    "Decision Tree": accuracy_score(y_test, dt_pred),
    "Random Forest": accuracy_score(y_test, rf_pred)
}

comparison_df = pd.DataFrame.from_dict(
    results, orient='index', columns=['Accuracy']
)

print("Model Comparision")
print(comparison_df)

plt.figure(figsize=(8, 5))
plt.bar(comparison_df.index, comparison_df['Accuracy'], color='skyblue')
plt.title("Model Accuracy Comparison (with Over-Sampling)", fontsize=14)
plt.xticks(rotation=30, ha='right')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.show()

# ## Evaluation of Best Model


from sklearn.metrics import confusion_matrix, classification_report

print("Random Forest - Initial Evaluation")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))
print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

# ## Hyperparameter Tuning with GridSearchCV


from sklearn.model_selection import GridSearchCV

rf_grid = RandomForestClassifier(
    random_state=42,
    class_weight='balanced'
)

param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [15, 20, 25, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(
    estimator=rf_grid,
    param_grid=param_grid,
    cv=5,
    scoring='f1_macro',  # Changed to f1_macro for better handling of imbalanced data
    n_jobs=-1,
    verbose=1
)

print("Hyperparameter Tuning")
print("Starting GridSearchCV (this may take a few minutes)...")
grid_search.fit(X_train_balanced, y_train_balanced)

# ### Train Final Random Forest Model with Best Parameters


final_rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=25,
    class_weight='balanced',
    random_state=42
)

# Train on balanced data
final_rf.fit(X_train_balanced, y_train_balanced)

# Predict on test data
final_rf_pred = final_rf.predict(X_test)

# Evaluation
print("\nFinal Random Forest Performance")
print("Accuracy:", accuracy_score(y_test, final_rf_pred))
print("\nClassification Report:")
print(classification_report(y_test, final_rf_pred))

# ### Plot Confusion Matrix


from sklearn.metrics import ConfusionMatrixDisplay

cm = confusion_matrix(y_test, final_rf_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['Average', 'Excellent', 'Good']
)

fig, ax = plt.subplots(figsize=(7, 5))
disp.plot(ax=ax, cmap='Blues', values_format='d')
plt.title("Confusion Matrix – Final Random Forest Model")
plt.tight_layout()
plt.show()

# ### Save Final Model in Pickle file


with open("rf_model.pkl", "wb") as f:
    pickle.dump(final_rf, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully!")

