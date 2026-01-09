import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist, pdist
import warnings
warnings.filterwarnings('ignore')

# =================================================================
# INDICE DE DUNN
# =================================================================
def dunn_index(X, labels):
    unique_labels = [c for c in np.unique(labels) if c != -1]
    if len(unique_labels) < 2:
        return None

    intra_dists = []
    for c in unique_labels:
        cluster_points = X[labels == c]
        if len(cluster_points) > 1:
            intra_dists.append(np.max(pdist(cluster_points)))

    if not intra_dists:
        return None

    max_intra = np.max(intra_dists)

    inter_dists = []
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            ci = X[labels == unique_labels[i]]
            cj = X[labels == unique_labels[j]]
            inter_dists.append(np.min(cdist(ci, cj)))

    if not inter_dists:
        return None

    min_inter = np.min(inter_dists)
    return min_inter / max_intra


# =================================================================
# 1. CHARGEMENT
# =================================================================
print(" CHARGEMENT DES DONNÉES")
df = pd.read_csv('dataset_unbalanced.csv')
print(f"Dimensions: {df.shape}")

# =================================================================
# 3. PRÉTRAITEMENT
# =================================================================
X = df.drop('FIRE', axis=1)
y = df['FIRE']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_np = X_scaled


# =================================================================
# 4. CLUSTERING FROM SCRATCH
# =================================================================

# -------------------------------------------------
# KMEANS
# -------------------------------------------------
class KMeansScratch:
    def __init__(self, n_clusters=2, max_iter=300, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        np.random.seed(random_state)

    def fit_predict(self, X):
        n_samples = X.shape[0]
        centroids = X[np.random.choice(n_samples, self.n_clusters, replace=False)]

        for _ in range(self.max_iter):
            distances = cdist(X, centroids)
            labels = np.argmin(distances, axis=1)

            new_centroids = np.array([
                X[labels == k].mean(axis=0) for k in range(self.n_clusters)
            ])

            if np.linalg.norm(centroids - new_centroids) < self.tol:
                break

            centroids = new_centroids

        return labels


# -------------------------------------------------
# DBSCAN
# -------------------------------------------------
class DBSCANScratch:
    def __init__(self, eps=0.4, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def fit_predict(self, X):
        n = X.shape[0]
        labels = np.full(n, -1)
        visited = np.zeros(n, bool)
        cluster_id = 0
        distances = cdist(X, X)

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            neighbors = np.where(distances[i] <= self.eps)[0]

            if len(neighbors) < self.min_samples:
                labels[i] = -1
            else:
                labels[i] = cluster_id
                self.expand_cluster(i, neighbors, labels, visited, cluster_id, distances)
                cluster_id += 1

        return labels

    def expand_cluster(self, i, neighbors, labels, visited, cluster_id, distances):
        j = 0
        while j < len(neighbors):
            p = neighbors[j]
            if not visited[p]:
                visited[p] = True
                p_neighbors = np.where(distances[p] <= self.eps)[0]
                if len(p_neighbors) >= self.min_samples:
                    neighbors = np.unique(np.concatenate((neighbors, p_neighbors)))
            if labels[p] == -1:
                labels[p] = cluster_id
            j += 1


# -------------------------------------------------
# CLARA FROM SCRATCH
# -------------------------------------------------
class CLARAScratch:
    def __init__(self, n_clusters=2, num_samples=5, sample_size=100, random_state=42):
        self.n_clusters = n_clusters
        self.num_samples = num_samples
        self.sample_size = sample_size
        np.random.seed(random_state)

    def fit_predict(self, X):
        n = X.shape[0]
        best_medoids = None
        best_cost = np.inf
        
        for _ in range(self.num_samples):
            # Échantillonnage aléatoire
            sample_indices = np.random.choice(n, min(self.sample_size, n), replace=False)
            X_sample = X[sample_indices]
            
            # Appliquer k-medoids sur l'échantillon
            medoids_indices = self.k_medoids_on_sample(X_sample)
            
            # Trouver les médoïdes correspondants dans le dataset complet
            medoids = sample_indices[medoids_indices]
            
            # Calculer le coût sur le dataset complet
            cost = self.compute_cost(X, X[medoids])
            
            if cost < best_cost:
                best_cost = cost
                best_medoids = medoids
        
        # Assigner tous les points aux meilleurs médoïdes
        distances = cdist(X, X[best_medoids])
        return np.argmin(distances, axis=1)

    def k_medoids_on_sample(self, X_sample):
        n_sample = X_sample.shape[0]
        medoids_indices = np.random.choice(n_sample, self.n_clusters, replace=False)
        
        for _ in range(10):  # Nombre d'itérations limité
            # Étape d'assignation
            distances = cdist(X_sample, X_sample[medoids_indices])
            labels = np.argmin(distances, axis=1)
            
            # Étape de mise à jour des médoïdes
            new_medoids_indices = medoids_indices.copy()
            for i in range(self.n_clusters):
                cluster_points = X_sample[labels == i]
                if len(cluster_points) > 0:
                    # Trouver le point qui minimise la distance totale
                    distances_in_cluster = cdist(cluster_points, cluster_points)
                    total_distances = distances_in_cluster.sum(axis=1)
                    best_idx = np.argmin(total_distances)
                    
                    # Trouver l'indice global du meilleur point
                    cluster_indices = np.where(labels == i)[0]
                    new_medoids_indices[i] = cluster_indices[best_idx]
            
            if np.array_equal(medoids_indices, new_medoids_indices):
                break
            medoids_indices = new_medoids_indices
        
        return medoids_indices

    def compute_cost(self, X, medoids):
        distances = cdist(X, medoids)
        return np.sum(np.min(distances, axis=1))


# =================================================================
# 5. APPLICATION
# =================================================================
models = {
    "KMeans": KMeansScratch(2),
    "DBSCAN": DBSCANScratch(),
    "CLARA": CLARAScratch(2)
}

results = {}
for name, model in models.items():
    print(f" Application de {name}")
    results[name] = model.fit_predict(X_np)


# =================================================================
# 6. ÉVALUATION
# =================================================================
print("\n ÉVALUATION DES RÉSULTATS")

print("\n AVEC TARGET (FIRE):")
print("-" * 40)

for name, labels in results.items():
    ari = adjusted_rand_score(y, labels)
    nmi = normalized_mutual_info_score(y, labels)

    print(f"\n{name}:")
    print(f"  ARI: {ari:.4f}")
    print(f"  NMI: {nmi:.4f}")

print("\n SANS TARGET:")
print("-" * 40)

for name, labels in results.items():
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters > 1 and len(X_np) > n_clusters:
        silhouette = silhouette_score(X_np, labels)
        ch = calinski_harabasz_score(X_np, labels)
        db = davies_bouldin_score(X_np, labels)
        dunn = dunn_index(X_np, labels)
    else:
        silhouette, ch, db, dunn = -1, -1, np.inf, None

    noise = np.sum(labels == -1)

    print(f"\n{name}:")
    print(f"  Clusters: {n_clusters}")
    print(f"  Points de bruit: {noise}")
    print(f"  Silhouette Score: {silhouette:.4f}")
    print(f"  Calinski-Harabasz: {ch:.4f}")
    print(f"  Davies-Bouldin: {db:.4f}")
    if dunn is not None:
        print(f"  Dunn Index: {dunn:.4f}")


# =================================================================
# 7. VISUALISATION
# =================================================================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_np)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

axes[0,0].scatter(X_pca[:,0], X_pca[:,1], c=y)
axes[0,0].set_title("Vraies classes")

axes[0,1].scatter(X_pca[:,0], X_pca[:,1], c=results["KMeans"])
axes[0,1].set_title("KMeans")

axes[1,0].scatter(X_pca[:,0], X_pca[:,1], c=results["DBSCAN"])
axes[1,0].set_title("DBSCAN")

axes[1,1].scatter(X_pca[:,0], X_pca[:,1], c=results["CLARA"])
axes[1,1].set_title("CLARA")

plt.tight_layout()
plt.show()






""""

# =========================================================================================================================================================================
# =================================================================SCikit-LEARN IMPLEMENTATION =================================================================
#=========================================================================================================================================================================



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist, pdist
import warnings
warnings.filterwarnings('ignore')

# Import pour CLARA (PyClustering)
try:
    from pyclustering.cluster.clara import clara as CLARA_algorithm
    CLARA_AVAILABLE = True
except ImportError:
    print("  PyClustering n'est pas installé. CLARA ne sera pas disponible.")
    print("   Installez-le avec: pip install pyclustering")
    CLARA_AVAILABLE = False

# Import scikit-learn clustering algorithms
from sklearn.cluster import KMeans, DBSCAN
from sklearn_extra.cluster import KMedoids

# =================================================================
# INDICE DE DUNN (même fonction que votre code)
# =================================================================
def dunn_index(X, labels):
    unique_labels = [c for c in np.unique(labels) if c != -1]
    if len(unique_labels) < 2:
        return None

    intra_dists = []
    for c in unique_labels:
        cluster_points = X[labels == c]
        if len(cluster_points) > 1:
            intra_dists.append(np.max(pdist(cluster_points)))

    if not intra_dists:
        return None

    max_intra = np.max(intra_dists)

    inter_dists = []
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            ci = X[labels == unique_labels[i]]
            cj = X[labels == unique_labels[j]]
            inter_dists.append(np.min(cdist(ci, cj)))

    if not inter_dists:
        return None

    min_inter = np.min(inter_dists)
    return min_inter / max_intra

# =================================================================
# 1. CHARGEMENT
# =================================================================
print(" CHARGEMENT DES DONNÉES")
print("=" * 50)

df = pd.read_csv('dataset_unbalanced.csv')
print(f"Dimensions: {df.shape}")
print(f"\nAperçu des données:")
print(df.head())

# =================================================================
# 2. PRÉTRAITEMENT
# =================================================================
print("\n PRÉTRAITEMENT")
print("=" * 50)

X = df.drop('FIRE', axis=1)
y = df['FIRE']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_np = X_scaled

print(f"Données normalisées - shape: {X_np.shape}")

# =================================================================
# 3. APPLICATION DES MODÈLES (scikit-learn)
# =================================================================
print("\n🔬 APPLICATION DES MODÈLES DE CLUSTERING")
print("=" * 50)

results = {}

# -------------------------------------------------
# 3.1 K-MEANS (scikit-learn)
# -------------------------------------------------
print("\n1. K-Means (scikit-learn)...")
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_np)
results["KMeans"] = kmeans_labels
print(f"   ✓ Terminé - {len(np.unique(kmeans_labels))} clusters")

# -------------------------------------------------
# 3.2 DBSCAN (scikit-learn)
# -------------------------------------------------
print("\n2. DBSCAN (scikit-learn)...")
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_np)
results["DBSCAN"] = dbscan_labels
n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
print(f"   ✓ Terminé - {n_clusters_dbscan} clusters, bruit: {np.sum(dbscan_labels == -1)} points")

# -------------------------------------------------
# 3.3 CLARA (PyClustering)
# -------------------------------------------------
if CLARA_AVAILABLE:
    print("\n3. CLARA (PyClustering)...")
    try:
        # Créer l'instance CLARA
        claras_instance = CLARA_algorithm(data=X_np.tolist(), number_clusters=2, num_samples=5, sample_size=100)
        
        # Exécuter le clustering
        claras_instance.process()
        
        # Obtenir les résultats
        claras_clusters = claras_instance.get_clusters()
        claras_medoids = claras_instance.get_medoids()
        
        # Convertir en labels (comme scikit-learn)
        claras_labels = np.full(X_np.shape[0], -1)
        for cluster_idx, cluster_points in enumerate(claras_clusters):
            for point_idx in cluster_points:
                claras_labels[point_idx] = cluster_idx
        
        results["CLARA"] = claras_labels
        print(f"   ✓ Terminé - {len(np.unique(claras_labels))} clusters")
        
    except Exception as e:
        print(f"   ✗ Erreur avec CLARA: {e}")
        results["CLARA"] = None


print("\n Tous les modèles appliqués")

# =================================================================
# 4. ÉVALUATION DES RÉSULTATS
# =================================================================
print("\n ÉVALUATION DES RÉSULTATS")
print("=" * 80)

print("\n AVEC TARGET (FIRE):")
print("-" * 40)

for name, labels in results.items():
    if labels is None:
        print(f"\n{name}: Non disponible")
        continue
    
    ari = adjusted_rand_score(y, labels)
    nmi = normalized_mutual_info_score(y, labels)

    print(f"\n{name}:")
    print(f"  ARI: {ari:.4f}")
    print(f"  NMI: {nmi:.4f}")

print("\n SANS TARGET:")
print("-" * 40)

for name, labels in results.items():
    if labels is None:
        print(f"\n{name}: Non disponible")
        continue
    
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Vérifier que nous avons au moins 2 clusters pour les métriques
    valid_for_metrics = (n_clusters > 1 and 
                         len(X_np) > n_clusters and
                         not (n_clusters == 2 and -1 in unique_labels and len(unique_labels) == 2))

    if valid_for_metrics:
        try:
            silhouette = silhouette_score(X_np, labels)
        except:
            silhouette = -1
        
        try:
            ch = calinski_harabasz_score(X_np, labels)
        except:
            ch = -1
        
        try:
            db = davies_bouldin_score(X_np, labels)
        except:
            db = np.inf
        
        dunn = dunn_index(X_np, labels)
    else:
        silhouette, ch, db, dunn = -1, -1, np.inf, None

    noise = np.sum(labels == -1)

    print(f"\n{name}:")
    print(f"  Clusters: {n_clusters}")
    print(f"  Points de bruit: {noise}")
    print(f"  Silhouette Score: {silhouette:.4f}")
    print(f"  Calinski-Harabasz: {ch:.4f}")
    print(f"  Davies-Bouldin: {db:.4f}")
    if dunn is not None:
        print(f"  Dunn Index: {dunn:.4f}")

# =================================================================
# 5. VISUALISATION AVEC PCA
# =================================================================
print("\n VISUALISATION DES CLUSTERS (PCA)")
print("=" * 80)

# Appliquer PCA pour réduire à 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_np)

print(f"Variance expliquée par les composantes PCA: {pca.explained_variance_ratio_.sum():.2%}")

# Créer la figure avec les subplots
n_models = len(results) + 1  # +1 pour les vraies classes
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

# Configuration des couleurs
colors_true = plt.cm.Set1(y.astype(int))
color_palettes = {
    'KMeans': plt.cm.tab10,
    'DBSCAN': plt.cm.Set2,
    'CLARA': plt.cm.Set3
}

# Plot 1: Vraies classes
ax = axes[0]
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', s=30, alpha=0.7)
ax.set_title("Vraies classes (FIRE)", fontsize=14, fontweight='bold')
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
ax.grid(True, alpha=0.3)

# Ajouter une légende pour les vraies classes
handles_true = [plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor='blue', markersize=8, label='Non-Feu (0)'),
               plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor='red', markersize=8, label='Feu (1)')]
ax.legend(handles=handles_true, loc='upper right')

# Plots 2-4: Clusters des modèles
model_names = list(results.keys())
for i, model_name in enumerate(model_names, 1):
    ax = axes[i]
    labels = results[model_name]
    
    if labels is None:
        ax.text(0.5, 0.5, f"{model_name}\nNon disponible", 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(f"{model_name}", fontsize=14, fontweight='bold')
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.grid(True, alpha=0.3)
        continue
    
    # Utiliser une palette de couleurs appropriée
    cmap = color_palettes.get(model_name, plt.cm.tab10)
    
    # Gérer les points de bruit (-1) séparément
    noise_mask = labels == -1
    valid_mask = ~noise_mask
    
    if np.any(noise_mask):
        # Afficher les points valides avec couleurs
        scatter_valid = ax.scatter(X_pca[valid_mask, 0], X_pca[valid_mask, 1], 
                                  c=labels[valid_mask], cmap=cmap, s=30, alpha=0.7)
        # Afficher les points de bruit en gris
        scatter_noise = ax.scatter(X_pca[noise_mask, 0], X_pca[noise_mask, 1], 
                                  c='gray', s=20, alpha=0.5, marker='x')
        
        # Créer une légende qui inclut le bruit
        unique_labels_valid = np.unique(labels[valid_mask])
        handles = []
        for label in unique_labels_valid:
            color = cmap(label / max(1, len(unique_labels_valid) - 1))
            handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=color, markersize=8, 
                                     label=f'Cluster {label}'))
        
        # Ajouter le bruit à la légende
        if np.sum(noise_mask) > 0:
            handles.append(plt.Line2D([0], [0], marker='x', color='gray', 
                                     markersize=8, label=f'Bruit ({np.sum(noise_mask)} pts)'))
        
        ax.legend(handles=handles, loc='upper right', fontsize=9)
        
    else:
        # Pas de bruit, afficher normalement
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                            c=labels, cmap=cmap, s=30, alpha=0.7)
        
        # Créer une légende
        unique_labels = np.unique(labels)
        if len(unique_labels) <= 10:  # Limiter la légende à 10 clusters max
            handles = []
            for label in unique_labels:
                color = cmap(label / max(1, len(unique_labels) - 1))
                handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor=color, markersize=8, 
                                         label=f'Cluster {label}'))
            ax.legend(handles=handles, loc='upper right', fontsize=9)
    
    ax.set_title(f"{model_name}", fontsize=14, fontweight='bold')
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.grid(True, alpha=0.3)

# Ajuster l'espacement
plt.tight_layout()

# Sauvegarder et afficher
plt.savefig('clustering_results_pca.png', dpi=300, bbox_inches='tight')
plt.show()



print("\n Analyse terminée!")

"""