import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

# Neural Network Model Definition
class AttributeClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AttributeClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

# Load merged dataset
data_path = "data/unique_products_with_attributes.csv" 
data = pd.read_csv(data_path)

# Plot frequencies of the categoric variables
cat_columns = ['des_color', 'des_sex', 'des_age', 'des_line', 
               'des_fabric', 'des_product_category', 'des_product_aggregated_family',
               'des_product_family', 'des_product_type']

for col in cat_columns:
    if col == 'des_color' or col=='des_product_type':
        print(f"\nFrecuencias para la columna '{col}':\n")
        tabla_frecuencias = data[col].value_counts().reset_index()
        tabla_frecuencias.columns = [col, 'Frecuencia']
        print(tabla_frecuencias)
    else:
        print(f"\nFrecuencias de {col}:")
        print(data[col].value_counts())

        plt.figure(figsize=(10, 5))
        sns.countplot(data=data, y=col, order=data[col].value_counts().index, palette='viridis')
        plt.title(f"Distribución de {col}")

# Unique predictive cols
target_col = 'des_value'
predictive_cols = ['silhouette_type', 'neck_lapel_type', 'woven_structure',
                   'knit_structure', 'heel_shape_type', 'length_type',
                   'sleeve_length_type', 'toecap_type', 'waist_type',
                   'closure_placement', 'cane_height_type']

for col in predictive_cols:
    unique_values = data[col].nunique()
    print(f"Cardinalidad de {col}: {unique_values}")

# Select most important features
features = ["des_sex", "des_age", "des_line", "des_fabric", "des_product_family", "des_product_type"]

unique_attributes = ['silhouette_type', 'neck_lapel_type', 'woven_structure',
                    'knit_structure', 'heel_shape_type', 'length_type',
                    'sleeve_length_type', 'toecap_type', 'waist_type',
                    'closure_placement', 'cane_height_type']

# Initialize label encoders for each column
label_encoders = {}
mappings = {}

# Encode features and attributes using LabelEncoder
for col in features + unique_attributes:
    le = LabelEncoder()
    original_data = data[col]
    encoded_data = le.fit_transform(original_data)
    data[col] = encoded_data
    label_encoders[col] = le
    
    # Create mapping DataFrame
    df_i = pd.DataFrame({
        'original': le.classes_,
        'encoding': np.arange(len(le.classes_))
    })
    print(f"\nMapping for {col}:")
    print(df_i)
    mappings[col] = df_i

# Save mappings and label encoders
with open(f"pretrained_models/mappings_nn.pkl", "wb") as f:
    pickle.dump(mappings, f)
with open(f"pretrained_models/label_encoders_nn.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

# Split test and train models 
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

# Setup for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_size = 128
num_epochs = 10
batch_size = 32
learning_rate = 0.001

# Dictionary to store models and results
models = {}
results = []
all_y_true = []
all_y_pred = []

# Train a neural network for each attribute
for attribute in unique_attributes:
    print(f"\nTraining neural network for attribute: {attribute}")
    
    # Prepare data
    X_train = train_data[features].values
    y_train = train_data[attribute].values
    X_test = test_data[features].values
    y_test = test_data[attribute].values
    
    # Get number of classes for this attribute using the label encoder
    n_classes = len(label_encoders[attribute].classes_)
    print(f"Number of classes for {attribute}: {n_classes}")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    
    # Create DataLoader
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    input_size = len(features)
    model = AttributeClassifier(input_size, hidden_size, n_classes).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {total_loss/len(train_loader):.4f}')
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        X_test_tensor = X_test_tensor.to(device)
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs.data, 1)
        y_pred = predicted.cpu().numpy()
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {attribute}: {accuracy:.4f}")
    
    # Store results
    models[attribute] = model
    torch.save(model.state_dict(), f"pretrained_models/nn_model_{attribute}.pth")
    results.append(accuracy)
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)
    
    # Map predictions back to labels using label encoder
    y_pred_labeled = label_encoders[attribute].inverse_transform(y_pred)
    test_data[f"{attribute}_predicted"] = y_pred_labeled

# Calculate total accuracy
total_accuracy = accuracy_score(all_y_true, all_y_pred)
print(f"\nTotal accuracy of the model: {total_accuracy:.4f}")

# Load and process real test data
data_path = "data/test_data.csv"
real_test_df = pd.read_csv(data_path)

# Encode features in test data using label encoders
for feature in features:
    le = label_encoders[feature]
    # Handle unknown values by setting them to a default value
    real_test_df[feature] = real_test_df[feature].map(
        lambda x: le.transform([x])[0] if x in le.classes_ else -1
    )

# Initialize prediction column
real_test_df["des_value"] = ["" for _ in range(len(real_test_df))]

# Make predictions for each attribute
for attribute in unique_attributes:
    # Get rows for current attribute
    mask = real_test_df["attribute_name"] == attribute
    if not mask.any():
        continue
        
    # Prepare input data
    X = real_test_df.loc[mask, features].values
    X_tensor = torch.FloatTensor(X).to(device)
    
    # Get predictions
    model = models[attribute]
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs.data, 1)
        y_pred = predicted.cpu().numpy()
    
    # Map back to labels using label encoder
    y_pred_labeled = label_encoders[attribute].inverse_transform(y_pred)
    
    # Store predictions
    real_test_df.loc[mask, "des_value"] = y_pred_labeled

# Save predictions
real_test_df[["test_id", "des_value"]].to_csv("test1.csv", index=False)