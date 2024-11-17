import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Definición del Dataset personalizado
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings.astype(np.float32)
        self.labels = labels
    def __len__(self):
        return len(self.embeddings)
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# Definición del Modelo
class EmbeddingClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(EmbeddingClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        return out

def main(args):
    # Cargar los datasets
    embeddings_df = pd.read_csv('image_random_vectors.csv')  # Reemplaza con el nombre real del archivo
    attributes_df = pd.read_csv('data/unique_products_with_attributes.csv')  # Reemplaza con el nombre real del archivo

    # Filtrar atributos INVALID
    attributes_df = attributes_df[attributes_df[args.attribute] != 'INVALID']

    # Unir datasets
    merged_df = pd.merge(embeddings_df, attributes_df, left_on='name', right_on='cod_modelo_color')

    # Codificar las etiquetas
    label_encoder = LabelEncoder()
    merged_df['label'] = label_encoder.fit_transform(merged_df[args.attribute])

    # Extraer embeddings y etiquetas
    embedding_columns = [col for col in embeddings_df.columns if col.startswith('dim_')]
    embeddings = merged_df[embedding_columns].values
    labels = merged_df['label'].values

    # División de datos
    X_train, X_val, y_train, y_val = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Crear datasets y dataloaders
    train_dataset = EmbeddingDataset(X_train, y_train)
    val_dataset = EmbeddingDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Definir el modelo, criterio y optimizador
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EmbeddingClassifier(input_size=embeddings.shape[1], num_classes=len(label_encoder.classes_))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Entrenamiento
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for embeddings_batch, labels_batch in train_loader:
            embeddings_batch = embeddings_batch.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()
            outputs = model(embeddings_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * embeddings_batch.size(0)
        epoch_loss = running_loss / len(train_dataset)

        # Validación
        model.eval()
        val_running_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for embeddings_batch, labels_batch in val_loader:
                embeddings_batch = embeddings_batch.to(device)
                labels_batch = labels_batch.to(device)

                outputs = model(embeddings_batch)
                loss = criterion(outputs, labels_batch)

                val_running_loss += loss.item() * embeddings_batch.size(0)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels_batch.cpu().numpy())
        val_epoch_loss = val_running_loss / len(val_dataset)
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

        print(f'Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}, Val Acc: {accuracy:.4f}')

    # Reporte de clasificación
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))

    # Matriz de confusión
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    # Función de predicción
    def predict(embedding):
        model.eval()
        with torch.no_grad():
            embedding_tensor = torch.tensor(embedding, dtype=torch.float32).to(device)
            output = model(embedding_tensor.unsqueeze(0))
            _, pred = torch.max(output, 1)
            return label_encoder.inverse_transform(pred.cpu().numpy())[0]

    # Ejemplo de uso de la función de predicción
    sample_embedding = X_val[0]
    predicted_label = predict(sample_embedding)
    print(f'\nPredicted label for the sample embedding: {predicted_label}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model to predict product attributes from embeddings.')
    parser.add_argument('--attribute', type=str, required=True, help='The target attribute to predict.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer.')

    args = parser.parse_args()
    main(args)
