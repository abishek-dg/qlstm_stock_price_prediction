# === Your Existing QLSTM Code ===
import os
import logging
import warnings
warnings.filterwarnings("ignore")

# === Setup Logger Safely and Correctly ===
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()

log_file_path = os.path.join(base_dir, "log_qlstm_4qb_2nl_64hs_100ep.txt")

logger = logging.getLogger("QLSTMLogger")
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

logger.info("===== QLSTM Script Started =====")

# === Create Folders for Outputs ===
os.makedirs("models_qlstm_4qb_2nl_64hs_100ep", exist_ok=True)
os.makedirs("images_qlstm_4qb_2nl_64hs_100ep", exist_ok=True)

# === Import Libraries with Logging ===
try:
    import os
    import torch
    import random
    import matplotlib
    import numpy as np
    import pandas as pd
    matplotlib.use('Agg')
    import torch.nn as nn
    import torch.optim as optim
    import matplotlib.pyplot as plt
    from qiskit import QuantumCircuit
    from sklearn.metrics import confusion_matrix
    from sklearn.preprocessing import MinMaxScaler
    from qiskit.visualization import circuit_drawer
    from qiskit.primitives import StatevectorEstimator
    from torch.utils.data import DataLoader, TensorDataset
    from qiskit_machine_learning.connectors import TorchConnector
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    from qiskit_machine_learning.neural_networks import EstimatorQNN
    from sklearn.metrics import precision_recall_curve, average_precision_score , roc_curve, auc ,mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
except ImportError as e:
    logger.error(f"ImportError: {e}")
    raise SystemExit(f"ImportError: {e}")

# === Check device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# === Set seed value ===
try:
    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except Exception as e:
    logger.error("The error caused in setting seed value {e}")

# === Load & Preprocess Dataset ===
try:
    df = pd.read_csv("apple.csv")
    df = df[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce").dropna()

    feature_cols = ["Open", "High", "Low", "Volume"]
    target_col = ["Close"]

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    X_scaled = feature_scaler.fit_transform(df[feature_cols])
    y_scaled = target_scaler.fit_transform(df[target_col])

    X_data = X_scaled
    y_data = y_scaled

    def create_sequences(X_data, y_data, seq_length):
        X, y = [], []
        for i in range(len(X_data) - seq_length):
            X.append(X_data[i:i + seq_length])
            y.append(y_data[i + seq_length])
        return np.array(X), np.array(y)

    seq_length = 60
    X, y = create_sequences(X_data, y_data, seq_length)

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)  # send to GPU/CPU
    y_tensor = torch.tensor(y.reshape(-1, 1), dtype=torch.float32).to(device)

    train_size = int(0.8 * len(X_tensor))
    X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
    y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

except Exception as e:
    logger.error(f"Error in dataset preparation: {e}")
    raise SystemExit(f"Dataset preparation failed: {e}")

def plot_train_test_split(X_train, X_test, y_train, y_test,
                          base_dir=".", image_dir="images_qlstm_4qb_2nl_64hs_100ep",
                          filename="train_test_split_plot_qlstm_4qb_2nl_64hs_100ep.png"):
    try:
        # Convert to CPU numpy
        y_train_np = y_train.detach().cpu().numpy()
        y_test_np = y_test.detach().cpu().numpy()

        # Build x-axis based on position (or from original datetime index)
        train_len = len(y_train_np)
        test_len = len(y_test_np)

        x_train_axis = list(range(train_len))
        x_test_axis = list(range(train_len, train_len + test_len))

        # Plotting
        plt.figure(figsize=(14, 5))
        plt.plot(x_train_axis, y_train_np, color='orange', label='TRAINING SET')
        plt.plot(x_test_axis, y_test_np, color='blue', label='TEST SET')
        plt.axvline(x=train_len, color='red', linestyle='--', label='Train/Test Split')

        plt.title("Train/Test Split Plot Using Input Tensors")
        plt.xlabel("Time Index")
        plt.ylabel("Target")
        plt.legend()
        plt.tight_layout()

        # Save
        save_path = os.path.join(base_dir, image_dir)
        os.makedirs(save_path, exist_ok=True)
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, dpi=300)
        plt.close()

        logger.info(f"Saved train-test plot to {full_path}")
    except Exception as e:
        logger.error(f"Error in plot_train_test_split: {e}")

plot_train_test_split(X_train, X_test, y_train, y_test)

try:
    # Quantum circuit setup
    num_qubits = 4
    feature_map = ZZFeatureMap(num_qubits, reps=2)
    ansatz = RealAmplitudes(num_qubits, reps=2, entanglement="full")
    qc = QuantumCircuit(num_qubits)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)

    estimator = StatevectorEstimator()
    qnn = EstimatorQNN(
            circuit=qc,
            estimator=estimator,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            input_gradients=True
        )
    quantum_layer = TorchConnector(qnn).to(device)

    try:
        qc_blocks = QuantumCircuit(num_qubits)
        qc_blocks.append(feature_map.to_gate(label="ZZFeatureMap"), range(num_qubits))
        qc_blocks.append(ansatz.to_gate(label="RealAmplitudes"), range(num_qubits))
        circuit_drawer(qc_blocks, output='mpl', filename=os.path.join(base_dir, "images_qlstm_4qb_2nl_64hs_100ep", "qlstm_architecture_blocks_4qb_2nl_64hs_100ep.png"))

        detailed_qc = QuantumCircuit(num_qubits)
        detailed_qc.compose(feature_map, inplace=True)
        detailed_qc.compose(ansatz, inplace=True)
        circuit_drawer(detailed_qc.decompose(), output='mpl', filename=os.path.join(base_dir, "images_qlstm_4qb_2nl_64hs_100ep", "featuremap_ansatz_internal_gates_qlstm_4qb_2nl_64hs_100ep.png"))

        logger.info("Saved updated quantum circuit diagrams (internal gates and block level).")
    except Exception as e:
        logger.error(f"Error saving quantum diagrams: {e}")

except Exception as e:
    logger.error(f"Error initializing QNN: {e}")

try:
    # Define QLSTM model
    # Define QLSTM model with Dropout
    class QLSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_qubits, quantum_layer):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
            self.dropout = nn.Dropout(p=0.2)  # Dropout after LSTM output
            self.fc2 = nn.Sequential(
                nn.Linear(hidden_size, num_qubits),
                nn.Tanh()
            )
            self.q_layer = quantum_layer
            self.fc_out = nn.Linear(1, 1)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            last_hidden = lstm_out[:, -1, :]  # Take last time step
            x = self.dropout(last_hidden)     # Apply dropout here
            x = self.fc2(x)
            x = self.q_layer(x)
            x = self.fc_out(x)
            return x

except  Exception as e:
    logger.error(f"The error occured in the QLSTM model creation {e}")

    # Instantiate model
hidden_size = 64
model = QLSTMModel(input_size=4, hidden_size=hidden_size, num_qubits=num_qubits, quantum_layer=quantum_layer).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Early stopping setup
early_stopping_patience = 5
best_loss = float('inf')
patience_counter = 0

try:
    # Draw neural network
    def draw_network(layer_sizes, layer_labels, save_path="images_qlstm_4qb_2nl_64hs_100ep/neural_network_qlstm_4qb_2nl_64hs_100ep.png"):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        x_positions = np.linspace(0.1, 0.9, len(layer_sizes))
        for i, (n_nodes, x) in enumerate(zip(layer_sizes, x_positions)):
            y_positions = [0.5] if n_nodes == 1 else np.linspace(0.15, 0.85, n_nodes)
            for y in y_positions:
                circle = plt.Circle((x, y), 0.02, fill=True)
                ax.add_artist(circle)
            ax.text(x, 0.92, layer_labels[i], ha='center', va='center', fontsize=10, weight='bold')
            if i < len(layer_sizes) - 1:
                next_n = layer_sizes[i + 1]
                next_x = x_positions[i + 1]
                next_y_positions = [0.5] if next_n == 1 else np.linspace(0.15, 0.85, next_n)
                for y1 in y_positions:
                    for y2 in next_y_positions:
                        ax.plot([x + 0.02, next_x - 0.02], [y1, y2], linewidth=0.5)
        ax.set_title("QLSTM NEURAL NETWORK", fontsize=12, pad=20)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

    # Visualize network
    fc2_out = model.fc2[0].out_features  # Output of the Linear layer in fc2
    qnn_out = model.fc_out.in_features   # Input to the final Linear layer (QNN output)

    layer_sizes = [seq_length, hidden_size, fc2_out, qnn_out, 1]
    layer_labels = [
        f"Input ({seq_length}×1)", 
        f"LSTM (2 Layers, {hidden_size}) + Dropout", 
        f"fc2 Linear → Tanh ({fc2_out})", 
        f"QNN Output ({qnn_out})", 
        "Final Output"
    ]

    draw_network(layer_sizes, layer_labels)
    logger.info("Saved updated QLSTM network architecture diagram.")

except Exception as e:
    logger.error(f"Error drawing architecture diagram: {e}") 

try:
    #creating the directary

    model_dir = "models_qlstm_4qb_2nl_64hs_100ep"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "qlstm_model_4qb_2nl_64hs_100ep.pt")

    # Training loop with early stopping
    epochs = 100
    train_losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            train_losses.append(avg_loss)
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info("Early stopping triggered.")
                break

    # Load best model after training
    model.load_state_dict(torch.load(model_path))

except  Exception as e:
    logger.error(f"The error caused in model saving and training : {e}")

try:
    # Drawing loss plot
    
    plt.switch_backend('Agg')
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, epochs + 1), train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss per Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(base_dir, "images_qlstm_4qb_2nl_64hs_100ep", "training_loss_4qb_2nl_64hs_100ep.png"), bbox_inches='tight', dpi=300)
    plt.close()
    logger.info("Saved training loss plot.")
except Exception as e:
    logger.error(f"Error saving loss plot: {e}")

try:
    # Prediction
    model.eval()
    predicted = []
    actual = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds = model(xb).cpu().numpy()
            predicted.extend(preds)
            actual.extend(yb.cpu().numpy())

    predicted_rescaled = target_scaler.inverse_transform(np.array(predicted))
    actual_rescaled = target_scaler.inverse_transform(np.array(actual))

    # Plot prediction
    os.makedirs("images_qlstm_4qb_2nl_64hs_100ep", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(actual_rescaled, label='Actual')
    plt.plot(predicted_rescaled, label='Predicted')
    plt.title('QLSTM Prediction - AAPL(Training data)')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig("images_qlstm_4qb_2nl_64hs_100ep/aapl_prediction_qlstm_4qb_2nl_64hs_100ep.png", bbox_inches='tight', dpi=300)
    plt.close()
    logger.info("Saved AAPL prediction plot.")
except Exception as e:
    logger.error(f"Error during evaluation or plotting: {e}")


# === Define Evaluation Function ===
def train_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)

    # Convert predictions to binary labels
    y_pred_labels = (y_pred.cpu().numpy() > 0.5).astype(int)
    y_test_labels = (y_test.cpu().numpy() > 0.5).astype(int)

    accuracy = round(accuracy_score(y_test_labels, y_pred_labels) * 100, 2)
    precision = round(precision_score(y_test_labels, y_pred_labels) * 100, 2)
    recall = round(recall_score(y_test_labels, y_pred_labels) * 100, 2)
    f1 = round(f1_score(y_test_labels, y_pred_labels) * 100, 2)
    roc_auc = round(roc_auc_score(y_test_labels, y_pred_labels) * 100, 2)
    mse = round(mean_squared_error(y_test, y_pred) * 100, 2)
    mae = round(mean_absolute_error(y_test, y_pred) * 100, 2)
    r2 = round(r2_score(y_test, y_pred) * 100, 2)
    explained_variance = round(explained_variance_score(y_test, y_pred)* 100, 2)
    rmse = round(np.sqrt(mse) * 100, 2)
    return accuracy, precision, recall, f1, roc_auc, y_test_labels, y_pred_labels, mse, mae, r2, explained_variance, rmse

# === Run Evaluation and Export ===
try:
    accuracy, precision, recall, f1, roc_auc, y_test_labels, y_pred_labels, mse, mae, r2, explained_variance, rmse= train_evaluate_model(model, X_train, X_test, y_train, y_test)

    eval_data = {
        'Title': ['qlstm_4qb_2nl_64hs_100ep'],
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1': [f1],
        'ROC_AUC': [roc_auc],
        'MSE': [mse],
        'MAE': [mae],
        'R2': [r2],
        'Explained_Variance': [explained_variance],
        'RMSE': [rmse]
    }

    eval_df = pd.DataFrame(eval_data)
    csv_path = os.path.join(base_dir, "qlstm_architecture_network_4qb_2nl_64hs_100ep.csv")
    eval_df.to_csv(csv_path, index=False)
    logger.info("Saved evaluation metrics to CSV.")


    try:
        def plot_confusion_matrix_matplotlib(cm, class_names, filename):
            try:
                fig, ax = plt.subplots(figsize=(6, 5))
                cax = ax.matshow(cm, cmap=plt.cm.Blues)
                plt.title("Confusion Matrix")
                fig.colorbar(cax)
                ax.set_xticklabels([''] + class_names)
                ax.set_yticklabels([''] + class_names)
                plt.xlabel('Predicted')
                plt.ylabel('True')

                for (i, j), val in np.ndenumerate(cm):
                    ax.text(j, i, f'{val}', ha='center', va='center')

                plt.savefig(filename, bbox_inches='tight', dpi=300)
                plt.close()
                logger.info(f"Saved confusion matrix plot to {filename}.")
            except Exception as e:
                logger.error(f"Error while plotting confusion matrix: {e}")
        cm = confusion_matrix(y_test_labels, y_pred_labels)
        plot_confusion_matrix_matplotlib(cm, ['Negative', 'Positive'], os.path.join(base_dir, "images_qlstm_4qb_2nl_64hs_100ep", "confusion_qlstm_4qb_2nl_64hs_100ep.png"))
    except Exception as e:
        logger.error(f"Error in creating Confusion Matrix {e}")

    # === Precision-Recall Curve Plot ===
    # === ROC and Precision-Recall Fixes ===
    try:
        model.eval()
        with torch.no_grad():
            y_pred_prob_tensor = model(X_test)
            y_pred_prob = y_pred_prob_tensor.detach().cpu().numpy().flatten()
            y_true = y_test.detach().cpu().numpy().flatten()

        # Binarize for classification metrics (e.g., classify up/down movement)
        median_val = np.median(y_true)
        y_true_binary = (y_true > median_val).astype(int)
        y_pred_binary = (y_pred_prob > median_val).astype(int)

        precision_vals, recall_vals, _ = precision_recall_curve(y_true_binary, y_pred_prob)
        avg_precision = average_precision_score(y_true_binary, y_pred_prob)

        plt.figure(figsize=(8, 6))
        plt.plot(recall_vals, precision_vals, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.text(0.2, 0.5, 'Avg Precision = {:.2f}'.format(avg_precision), fontsize=12)
        plt.savefig(os.path.join(base_dir, "images_qlstm_4qb_2nl_64hs_100ep", "precision_recall_curve_qlstm_4qb_2nl_64hs_100ep.png"), bbox_inches='tight', dpi=300)
        plt.close()
        logger.info("Saved Precision-Recall curve plot.")

        fpr, tpr, _ = roc_curve(y_true_binary, y_pred_prob)
        roc_auc_val = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc_val)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(base_dir, "images_qlstm_4qb_2nl_64hs_100ep", "roc_curve_qlstm_4qb_2nl_64hs_100ep"), bbox_inches='tight', dpi=300)
        plt.close()
        logger.info("Saved ROC curve plot.")
    except Exception as e:
        logger.error(f"Error while plotting PR/ROC curves: {e}")
except Exception as e:
    logger.error(f"Error in craeting csv file or in model evaluation {e}")

logger.info("Training part completed")


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------- Testing the model --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

try:
    seq_length = 60
    num_days_to_predict = 5

    # Load and scale dataset
    df = pd.read_csv("apple.csv")
    df = df[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce").dropna()
    feature_cols = ["Open", "High", "Low", "Volume"]
    target_col = ["Close"]

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    X_scaled = feature_scaler.fit_transform(df[feature_cols])
    y_scaled = target_scaler.fit_transform(df[target_col])

    X_input = X_scaled[-seq_length:].reshape(1, seq_length, len(feature_cols))
    X_input = torch.tensor(X_input, dtype=torch.float32).to(device)

    model.load_state_dict(torch.load("models_qlstm_4qb_2nl_64hs_100ep/qlstm_model_4qb_2nl_64hs_100ep.pt", map_location=device))
    model.eval()

    predicted_scaled = []
    input_seq = X_input.clone()

    with torch.no_grad():
        for _ in range(num_days_to_predict):
            next_val = model(input_seq)
            predicted_scaled.append(next_val.cpu().numpy().flatten()[0])
            new_input = torch.zeros((1, 1, len(feature_cols))).to(device)
            new_input[0, 0, -1] = next_val  # only last feature gets prediction
            input_seq = torch.cat([input_seq[:, 1:, :], new_input], dim=1)

    predicted_scaled = np.array(predicted_scaled).reshape(-1, 1)
    predicted_prices = target_scaler.inverse_transform(predicted_scaled)
    predicted_y = predicted_prices.flatten()
    predicted_x = range(seq_length, seq_length + num_days_to_predict)

    future_actual_df = pd.read_csv("future_actual.csv")
    future_actual = pd.to_numeric(future_actual_df["Close"], errors='coerce').dropna().values[:num_days_to_predict]

    last_days = df["Close"].values[-seq_length:]
    actual_full = np.concatenate((last_days, future_actual))

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(actual_full)), actual_full, label="Actual", color="blue")
    plt.plot(predicted_x, predicted_y, label="Predicted", color="green")
    for i, x in enumerate(predicted_x):
        if i < len(future_actual):
            plt.plot([x, x], [future_actual[i], predicted_y[i]], color="red", linestyle="dotted", linewidth=1)
    plt.axvline(x=seq_length, color='gray', linestyle='--', linewidth=1.5, label="Prediction Start")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.title(f"AAPL Price Prediction vs Actual (Next {num_days_to_predict} Days) - QLSTM")
    plt.legend()
    os.makedirs("images_qlstm_4qb_2nl_64hs_100ep", exist_ok=True)
    plt.savefig(f"images_qlstm_4qb_2nl_64hs_100ep/actual_vs_predicted_{num_days_to_predict}days.png", bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f" Saved: images_qlstm_4qb_2nl_64hs_100ep/actual_vs_predicted_{num_days_to_predict}days.png")

    plt.figure(figsize=(10, 5))
    plt.plot(range(seq_length + 1), list(last_days) + [last_days[-1]], label="Actual (Last Days)", color="blue")
    predicted_x = range(seq_length, seq_length + num_days_to_predict)
    plt.plot(predicted_x, predicted_y, label=f"Predicted (Next {num_days_to_predict} Days)", color="green")
    plt.axvline(x=seq_length, color='gray', linestyle='--', linewidth=1.5, label="Prediction Start")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.title(f"AAPL Price Prediction - Next {num_days_to_predict} Days (QLSTM)")
    plt.legend()
    plt.savefig(f"images_qlstm_4qb_2nl_64hs_100ep/predicted_only_{num_days_to_predict}days_aligned.png", bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f"Saved: images_qlstm_4qb_2nl_64hs_100ep/predicted_only_{num_days_to_predict}days_aligned.png")

except Exception as e:
    logger.info(f"Error at future prediction {e}")