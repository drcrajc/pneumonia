import os
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Function to map class names to binary labels
def map_class_to_binary(class_name):
    if class_name == 'unhealthy':
        return 1
    elif class_name == 'healthy':
        return 0
    else:
        raise ValueError("Unknown class name")

# Function to generate ROC plot from test results
def generate_roc_from_excel(excel_file, output_file):
    # Read Excel file
    df = pd.read_excel(excel_file)
    
    # Map true class and predicted class names to binary labels
    true_class = df['True Class'].apply(map_class_to_binary).astype(int)  # Convert to numeric and map to binary labels
    predicted_class = df['Predicted Class'].apply(map_class_to_binary).astype(int)  # Convert to numeric and map to binary labels
    
    # Calculate fpr, tpr, and auc
    fpr, tpr, _ = roc_curve(true_class, predicted_class)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve with adjusted font sizes and line thickness
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=16, fontweight='bold')  # Increase font size and make it bold
    plt.ylabel('True Positive Rate', fontsize=16, fontweight='bold')   # Increase font size and make it bold
    plt.title('ROC Curve', fontsize=20, fontweight='bold')             # Increase font size and make it bold
    plt.xticks(fontsize=14)  # Increase tick font size
    plt.yticks(fontsize=14)  # Increase tick font size
    plt.legend(loc='lower right', fontsize=14)
    plt.grid(True, linestyle='-', linewidth=2, alpha=0.5)  # Darker grid lines with increased thickness
    
    # Create directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.savefig(output_file, dpi=300)  # Save plot as PNG with 300 dpi
    plt.show()

# Path to the Excel file containing test results
test_results_excel = 'jawahar/xception.xlsx'

# Output file path for saving the plot
output_file = 'jawahar/rocplots/xception.png'

# Generate ROC curve from the test results Excel file and save it as a PNG file
generate_roc_from_excel(test_results_excel, output_file)
