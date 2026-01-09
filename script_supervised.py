
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import math
import random
import warnings
warnings.filterwarnings('ignore')

# =================================================================
# IMPLÉMENTATION FROM SCRATCH DES MODÈLES
# =================================================================

class KNNFromScratch:
    """Implémentation from scratch de K-Nearest Neighbors"""
    
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        self.X_train = X.values if isinstance(X, pd.DataFrame) else X
        self.y_train = y.values if isinstance(y, pd.Series) else y
        return self
    
    def _euclidean_distance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))
    
    def predict(self, X):
        X_test = X.values if isinstance(X, pd.DataFrame) else X
        predictions = []
        
        for test_point in X_test:
            distances = []
            
            # Calcul des distances avec tous les points d'entraînement
            for i, train_point in enumerate(self.X_train):
                dist = self._euclidean_distance(test_point, train_point)
                distances.append((dist, self.y_train[i]))
            
            # Trouver les k plus proches voisins
            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:self.k]
            
            # Vote majoritaire
            labels = [label for _, label in k_nearest]
            most_common = Counter(labels).most_common(1)[0][0]
            predictions.append(most_common)
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        X_test = X.values if isinstance(X, pd.DataFrame) else X
        probabilities = []
        
        for test_point in X_test:
            distances = []
            
            for i, train_point in enumerate(self.X_train):
                dist = self._euclidean_distance(test_point, train_point)
                distances.append((dist, self.y_train[i]))
            
            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:self.k]
            
            labels = [label for _, label in k_nearest]
            count_0 = labels.count(0)
            count_1 = labels.count(1)
            total = count_0 + count_1
            
            prob_0 = count_0 / total if total > 0 else 0.5
            prob_1 = count_1 / total if total > 0 else 0.5
            
            probabilities.append([prob_0, prob_1])
        
        return np.array(probabilities)

class DecisionTreeFromScratch:
    """Implémentation from scratch d'un arbre de décision"""
    
    def __init__(self, max_depth=10, min_samples_split=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
    
    def _entropy(self, y):
        if len(y) == 0:
            return 0
        p1 = np.sum(y) / len(y)
        p0 = 1 - p1
        if p0 == 0 or p1 == 0:
            return 0
        return -p0 * np.log2(p0) - p1 * np.log2(p1)
    
    def _information_gain(self, parent, left_child, right_child):
        weight_left = len(left_child) / len(parent)
        weight_right = len(right_child) / len(parent)
        gain = self._entropy(parent) - (weight_left * self._entropy(left_child) + weight_right * self._entropy(right_child))
        return gain
    
    def _best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            for threshold in unique_values:
                left_indices = feature_values <= threshold
                right_indices = feature_values > threshold
                
                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue
                
                gain = self._information_gain(y, y[left_indices], y[right_indices])
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Critères d'arrêt
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_classes == 1):
            leaf_value = np.bincount(y).argmax()
            return {'type': 'leaf', 'value': leaf_value, 'samples': n_samples}
        
        # Trouver la meilleure séparation
        best_feature, best_threshold, best_gain = self._best_split(X, y)
        
        if best_gain == -1:  # Aucune séparation utile trouvée
            leaf_value = np.bincount(y).argmax()
            return {'type': 'leaf', 'value': leaf_value, 'samples': n_samples}
        
        # Séparer les données
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold
        
        # Construire les sous-arbres
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        return {
            'type': 'node',
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree,
            'samples': n_samples,
            'gain': best_gain
        }
    
    def fit(self, X, y):
        X_data = X.values if isinstance(X, pd.DataFrame) else X
        y_data = y.values if isinstance(y, pd.Series) else y
        self.tree = self._build_tree(X_data, y_data)
        return self
    
    def _predict_single(self, x, tree):
        if tree['type'] == 'leaf':
            return tree['value']
        
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_single(x, tree['left'])
        else:
            return self._predict_single(x, tree['right'])
    
    def predict(self, X):
        X_data = X.values if isinstance(X, pd.DataFrame) else X
        predictions = [self._predict_single(x, self.tree) for x in X_data]
        return np.array(predictions)
    
    def predict_proba(self, X):
        # Pour simplifier, retourner des probabilités basées sur la prédiction
        predictions = self.predict(X)
        proba = np.zeros((len(predictions), 2))
        for i, pred in enumerate(predictions):
            proba[i, pred] = 1.0
        return proba

class RandomForestFromScratch:
    """Implémentation from scratch d'une forêt aléatoire"""
    
    def __init__(self, n_estimators=10, max_depth=10, min_samples_split=5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
    
    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]
    
    def fit(self, X, y):
        X_data = X.values if isinstance(X, pd.DataFrame) else X
        y_data = y.values if isinstance(y, pd.Series) else y
        
        self.trees = []
        for _ in range(self.n_estimators):
            # Échantillon bootstrap
            X_bootstrap, y_bootstrap = self._bootstrap_sample(X_data, y_data)
            
            # Créer et entraîner un arbre
            tree = DecisionTreeFromScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
        
        return self
    
    def predict(self, X):
        # Vote majoritaire de tous les arbres
        all_predictions = np.array([tree.predict(X) for tree in self.trees])
        # Prendre le mode pour chaque échantillon
        final_predictions = []
        for i in range(all_predictions.shape[1]):
            predictions = all_predictions[:, i]
            most_common = Counter(predictions).most_common(1)[0][0]
            final_predictions.append(most_common)
        
        return np.array(final_predictions)
    
    def predict_proba(self, X):
        # Moyenne des probabilités de tous les arbres
        all_proba = np.array([tree.predict_proba(X) for tree in self.trees])
        avg_proba = np.mean(all_proba, axis=0)
        return avg_proba

# =================================================================
# FONCTIONS UTILITAIRES POUR L'ÉVALUATION
# =================================================================

def accuracy_score(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def precision_score(y_true, y_pred):
    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    predicted_positives = np.sum(y_pred == 1)
    return true_positives / predicted_positives if predicted_positives > 0 else 0

def recall_score(y_true, y_pred):
    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    actual_positives = np.sum(y_true == 1)
    return true_positives / actual_positives if actual_positives > 0 else 0

def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return np.array([[tn, fp], [fn, tp]])

def roc_auc_score(y_true, y_proba):
    # Implémentation simplifiée de AUC-ROC
    thresholds = np.linspace(0, 1, 100)
    tprs = []
    fprs = []
    
    for threshold in thresholds:
        y_pred = (y_proba[:, 1] >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tprs.append(tpr)
        fprs.append(fpr)
    
    # Calcul de l'aire sous la courbe (méthode des trapèzes)
    auc = 0
    for i in range(1, len(thresholds)):
        auc += (fprs[i] - fprs[i-1]) * (tprs[i] + tprs[i-1]) / 2
    
    return abs(auc)



def train_test_split(X, y, test_size=0.2, random_state=42, stratify=None):
    np.random.seed(random_state)
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train = X.iloc[train_indices] if isinstance(X, pd.DataFrame) else X[train_indices]
    X_test = X.iloc[test_indices] if isinstance(X, pd.DataFrame) else X[test_indices]
    y_train = y.iloc[train_indices] if isinstance(y, pd.Series) else y[train_indices]
    y_test = y.iloc[test_indices] if isinstance(y, pd.Series) else y[test_indices]
    
    return X_train, X_test, y_train, y_test

# =================================================================
# 1. CHARGEMENT ET EXPLORATION DES DONNÉES
# =================================================================
print(" CHARGEMENT ET EXPLORATION DES DONNÉES")
print("=" * 50)

# Charger le dataset
df = pd.read_csv('dataset_balanced.csv')

print(f"Dimensions du dataset: {df.shape}")
print(f"\nColonnes: {list(df.columns)}")
print(f"\nAperçu des données:")
print(df.head())

# Vérifier la distribution de la target
print(f"\n DISTRIBUTION DE LA TARGET 'FIRE':")
fire_distribution = df['FIRE'].value_counts()
print(fire_distribution)
print(f"\nPourcentage de classe 1: {(fire_distribution[1]/len(df))*100:.2f}%")

# =================================================================
# 4. PRÉTRAITEMENT DES DONNÉES
# =================================================================
print("\n PRÉTRAITEMENT DES DONNÉES")
print("=" * 50)

# Copie du dataset pour le prétraitement
df_processed = df.copy()


# b. Séparation features/target
X = df_processed.drop('FIRE', axis=1)
y = df_processed['FIRE']

# c. Normalisation des features (manuellement)
print("Normalisation des features...")
def standard_scaler(X):
    X_scaled = X.copy()
    for col in X.columns:
        mean_val = X[col].mean()
        std_val = X[col].std()
        if std_val > 0:  # Éviter la division par zéro
            X_scaled[col] = (X[col] - mean_val) / std_val
        else:
            X_scaled[col] = 0
    return X_scaled

X_scaled = standard_scaler(X)

print(f"Features après normalisation: {X_scaled.shape}")

# d. Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"\n RÉPARTITION TRAIN/TEST:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
print(f"\nDistribution des classes dans y_train: {Counter(y_train)}")
print(f"Distribution des classes dans y_test: {Counter(y_test)}")

# =================================================================
# 5. SÉLECTION DES FEATURES IMPORTANTES
# =================================================================
print("\n SÉLECTION DES FEATURES IMPORTANTES")
print("=" * 50)

def select_important_features(X, y, n_features=15):
    """
    Sélection des features basée sur la corrélation avec la target
    """
    correlations = []
    for col in X.columns:
        if len(X[col].unique()) > 1:  # Éviter les variables constantes
            # Calcul manuel de la corrélation
            x_vals = X[col].values
            y_vals = y.values
            correlation = np.corrcoef(x_vals, y_vals)[0, 1]
            correlations.append((col, abs(correlation)))
    
    correlations.sort(key=lambda x: x[1], reverse=True)
    selected_features = [feature for feature, _ in correlations[:n_features]]
    
    print(f"\n TOP {n_features} FEATURES SÉLECTIONNÉES:")
    for feature, corr in correlations[:n_features]:
        print(f"  {feature}: {corr:.4f}")
    
    return selected_features

important_features = select_important_features(X_train, y_train, n_features=15)
X_train_selected = X_train[important_features]
X_test_selected = X_test[important_features]

# =================================================================
# 6. DÉFINITION ET ENTRAÎNEMENT DES MODÈLES FROM SCRATCH
# =================================================================
print("\n DÉFINITION ET ENTRAÎNEMENT DES MODÈLES FROM SCRATCH")
print("=" * 50)

# Définition des modèles from scratch
models = {
    "KNN (From Scratch)": KNNFromScratch(k=5),
    "Decision Tree (From Scratch)": DecisionTreeFromScratch(max_depth=10, min_samples_split=5),
    "Random Forest (From Scratch)": RandomForestFromScratch(n_estimators=10, max_depth=8, min_samples_split=5)
}

# Entraînement des modèles
optimized_models = {}

for name, model in models.items():
    print(f"\n🔧 Entraînement de {name}...")
    model.fit(X_train_selected, y_train)
    optimized_models[name] = model
    print(f" {name} entraîné")

# =================================================================
# 7. ÉVALUATION DES MODÈLES
# =================================================================
print("\n ÉVALUATION DES MODÈLES")
print("=" * 50)

def evaluate_model(model, name, X_test, y_test):
    """Évalue un modèle et retourne les métriques"""
    
    # Prédictions
    print(f"   Prédictions en cours...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Calcul des métriques
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'Modèle': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': auc_score,
        'Matrice Confusion': cm
    }

# Évaluation de tous les modèles
results = []
for name, model in optimized_models.items():
    print(f"Évaluation de {name}...")
    result = evaluate_model(model, name, X_test_selected, y_test)
    results.append(result)
    print(f" {name} évalué")

# Affichage des résultats
print("\n RÉSULTATS DÉTAILLÉS:")
print("=" * 80)
for result in results:
    print(f"\n{result['Modèle']}:")
    print(f"  Accuracy:  {result['Accuracy']:.4f}")
    print(f"  Precision: {result['Precision']:.4f}")
    print(f"  Recall:    {result['Recall']:.4f}")
    print(f"  F1-Score:  {result['F1-Score']:.4f}")
    print(f"  AUC-ROC:   {result['AUC-ROC']:.4f}")




""""

# =========================================================================================================================================================================
# =================================================================SCikit-LEARN IMPLEMENTATION =================================================================
#=========================================================================================================================================================================


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import des bibliothèques scikit-learn
from sklearn.model_selection import train_test_split, RandomizedSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           roc_curve, auc)

# =================================================================
# 1. CHARGEMENT DU DATASET
# =================================================================
print(" CHARGEMENT DU DATASET")
print("=" * 50)

# Charger le dataset
df = pd.read_csv('dataset_balanced.csv')

print(f"Dimensions du dataset: {df.shape}")
print(f"Colonnes: {list(df.columns)}")
print(f"\nDistribution de la target:")
print(df['FIRE'].value_counts())

# =================================================================
# 2. PRÉTRAITEMENT
# =================================================================
print("\n PRÉTRAITEMENT")
print("=" * 50)

# Séparation features/target
X = df.drop('FIRE', axis=1)
y = df['FIRE']

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# =================================================================
# 3. OPTIMISATION DES MODÈLES AVEC RANDOMIZEDSEARCHCV
# =================================================================
print("\n OPTIMISATION DES MODÈLES")
print("=" * 50)

# Modèles à optimiser
models = {
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

# Espaces d'hyperparamètres
param_dist = {
    'KNN': {
        'n_neighbors': [3, 5, 7, 10, 15, 20],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    },
    'Decision Tree': {
        'max_depth': [5, 8, 10, 12, 15, 20, None],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 6],
        'criterion': ['gini', 'entropy']
    },
    'Random Forest': {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [5, 8, 10, 12, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
}

# Stockage des résultats
results = []
optimized_models = {}

# Optimisation pour chaque modèle
for name, model in models.items():
    print(f"\nOptimisation de {name}...")
    
    # RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist[name],
        n_iter=15,
        cv=3,
        scoring='f1',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    # Entraînement
    random_search.fit(X_train, y_train)
    
    # Meilleur modèle
    best_model = random_search.best_estimator_
    optimized_models[name] = best_model
    
    # Prédictions
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Calcul des métriques
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    # Stocker les résultats
    results.append({
        'Modèle': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': auc_score,
        'Best Params': random_search.best_params_
    })
    
    print(f" {name} optimisé")

# =================================================================
# 4. AFFICHAGE DES MÉTRIQUES
# =================================================================
print("\n MÉTRIQUES DES MODÈLES OPTIMISÉS")
print("=" * 80)

# Créer et afficher le DataFrame
results_df = pd.DataFrame(results)
print(results_df[['Modèle', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']].to_string(index=False))

print("\n MEILLEURS HYPERPARAMÈTRES:")
for result in results:
    print(f"\n{result['Modèle']}:")
    for param, value in result['Best Params'].items():
        print(f"  {param}: {value}")

# =================================================================
# 5. COURBES ROC
# =================================================================
print("\n COURBES ROC")
print("=" * 50)

plt.figure(figsize=(10, 8))
colors = ['blue', 'green', 'red']

# Tracer chaque courbe ROC
for idx, (name, model) in enumerate(optimized_models.items()):
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=colors[idx], lw=2,
                label=f'{name} (AUC = {roc_auc:.3f})')

# Courbe de référence
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')

# Configuration
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de Faux Positifs')
plt.ylabel('Taux de Vrais Positifs')
plt.title('Courbes ROC - Modèles Optimisés')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# =================================================================
# 6. COURBES D'APPRENTISSAGE
# =================================================================
print("\n COURBES D'APPRENTISSAGE")
print("=" * 50)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Générer les courbes d'apprentissage pour chaque modèle
for idx, (name, model) in enumerate(optimized_models.items()):
    ax = axes[idx]
    
    # Calcul de la courbe d'apprentissage
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=3,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy', n_jobs=-1
    )
    
    # Calcul des moyennes
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Tracer
    ax.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, alpha=0.1, color='blue')
    ax.fill_between(train_sizes, test_mean - test_std,
                     test_mean + test_std, alpha=0.1, color='orange')
    
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Entraînement')
    ax.plot(train_sizes, test_mean, 'o-', color='orange', label='Validation')
    
    # Configuration
    ax.set_xlabel("Taille de l'échantillon")
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Courbe d\'apprentissage - {name}')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# =================================================================
# 7. MATRICES DE CONFUSION
# =================================================================
print("\n MATRICES DE CONFUSION")
print("=" * 50)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (name, model) in enumerate(optimized_models.items()):
    ax = axes[idx]
    
    # Prédictions
    y_pred = model.predict(X_test)
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    
    # Heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Non-Feu', 'Feu'],
                yticklabels=['Non-Feu', 'Feu'])
    
    ax.set_title(f'Matrice de confusion - {name}')
    ax.set_ylabel('Vraie étiquette')
    ax.set_xlabel('Étiquette prédite')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n Analyse terminée!")

"""