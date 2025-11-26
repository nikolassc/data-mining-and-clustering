import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def process_kmeans_advanced(input_file, k=5, n_init=500):
    # 1. Φόρτωση
    df = pd.read_csv(input_file, header=None, names=['x', 'y'])
    
    # 2. Scaling
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[['x', 'y']])
    df_scaled = pd.DataFrame(scaled_features, columns=['x', 'y'])

    # 3. K-Means με 50 αρχικοποιήσεις
    # Η βιβλιοθήκη αυτόματα κρατάει το καλύτερο αποτέλεσμα (μικρότερο inertia)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=n_init)
    df_scaled['cluster'] = kmeans.fit_predict(df_scaled[['x', 'y']])
    centroids = kmeans.cluster_centers_
    
    # Αρχικοποίηση λίστας για τα indices των outliers
    all_outliers_indices = []

    # 4. ΥΠΟΛΟΓΙΣΜΟΣ OUTLIERS ΑΝΑ CLUSTER (Per-Cluster Thresholding)
    print(f"--- Ανάλυση ανά Cluster για το αρχείο {input_file} ---")
    
    for cluster_id in range(k):
        # Παίρνουμε μόνο τα σημεία του συγκεκριμένου cluster
        cluster_data = df_scaled[df_scaled['cluster'] == cluster_id]
        indices = cluster_data.index
        
        # Υπολογισμός αποστάσεων από το κέντρο ΤΟΥ ΣΥΓΚΕΚΡΙΜΕΝΟΥ cluster
        centroid = centroids[cluster_id]
        distances = np.linalg.norm(cluster_data[['x', 'y']].values - centroid, axis=1)
        
        # Υπολογισμός στατιστικών ΜΟΝΟ για αυτό το cluster
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        # Τοπικό όριο για αυτό το cluster
        threshold = mean_dist + (3 * std_dist)
        
        # Βρίσκουμε ποια σημεία ξεπερνούν το τοπικό όριο
        outliers_mask = distances > threshold
        outliers_indices = indices[outliers_mask]
        all_outliers_indices.extend(outliers_indices)
        
        print(f"Cluster {cluster_id}: Mean Dist={mean_dist:.2f}, Std={std_dist:.2f}, Threshold={threshold:.2f} -> Outliers: {len(outliers_indices)}")

    # 5. Μαρκάρισμα στο αρχικό DataFrame
    df['label'] = 'Normal'
    df.loc[all_outliers_indices, 'label'] = 'Outlier'
    df['cluster'] = df_scaled['cluster']

    print(f"Συνολικά Outliers: {len(all_outliers_indices)}\n")

    # 6. Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot Normal
    normal = df[df['label'] == 'Normal']
    plt.scatter(normal['x'], normal['y'], c=normal['cluster'], cmap='viridis', s=15, alpha=0.6, label='Normal Points')
    
    # Plot Outliers
    outliers = df[df['label'] == 'Outlier']
    plt.scatter(outliers['x'], outliers['y'], c='red', marker='x', s=60, label='Outliers')
    
    plt.title(f'Optimized K-Means (n_init={n_init}) with Per-Cluster Outlier Detection\n({input_file})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Τρέξ' το και δες αν το μπλε cluster συμπεριφέρεται καλύτερα
process_kmeans_advanced('data/clean_data_a.csv', k=5, n_init=50)
process_kmeans_advanced('data/clean_data_b.csv', k=5, n_init=50)