import argparse
import json
import time
import random

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from deap import base, creator, tools

# ----------------------
# Argument parsing
# ----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--train_input", type=str)
parser.add_argument("--selected_features", type=str)
parser.add_argument("--baseline_metrics", type=str)
parser.add_argument("--ga_metrics", type=str)
args = parser.parse_args()

print("Loading train parquet:", args.train_input)
df = pd.read_parquet(args.train_input)

# Separate features/labels
X = df.drop(columns=["label", "image_id"])
y = df["label"].values

feature_names = list(X.columns)
n_features = X.shape[1]

# Train/validation split for baseline + GA
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ----------------------
# BASELINE FEATURE SELECTION
# ----------------------
print("Running baseline feature selection (VarianceThreshold)...")

# Simple filter based on variance (keep non-constant features)
vt = VarianceThreshold(threshold=0.0)
X_train_baseline = vt.fit_transform(X_train)
X_val_baseline = vt.transform(X_val)

baseline_mask = vt.get_support()
baseline_features = [f for f, keep in zip(feature_names, baseline_mask) if keep]

# Train quick model on baseline-selected features
baseline_clf = RandomForestClassifier(random_state=42)
baseline_clf.fit(X_train_baseline, y_train)
y_val_pred = baseline_clf.predict(X_val_baseline)
baseline_acc = accuracy_score(y_val, y_val_pred)

print("Baseline selected features:", baseline_features)
print("Baseline accuracy:", baseline_acc)

# Save baseline metrics
baseline_metrics = {
    "baseline_accuracy": float(baseline_acc),
    "baseline_num_features": len(baseline_features),
    "baseline_selected_features": baseline_features,
}
with open(args.baseline_metrics, "w") as f:
    json.dump(baseline_metrics, f)

# ----------------------
# GENETIC ALGORITHM FEATURE SELECTION (DEAP)
# ----------------------
print("Running Genetic Algorithm feature selection (DEAP)...")

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n_features)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evaluate_individual(individual):
    # Ensure at least one feature is selected
    if sum(individual) == 0:
        return 0.0,

    selected_idx = [i for i, bit in enumerate(individual) if bit == 1]

    Xtr_sel = X_train.iloc[:, selected_idx]
    Xval_sel = X_val.iloc[:, selected_idx]

    clf = RandomForestClassifier(random_state=42)
    clf.fit(Xtr_sel, y_train)
    y_pred = clf.predict(Xval_sel)
    acc = accuracy_score(y_val, y_pred)

    # Optional penalty: fewer features is better
    penalty = 0.001 * len(selected_idx)
    fitness = acc - penalty
    return fitness,


toolbox.register("evaluate", evaluate_individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

POP_SIZE = 20
N_GEN = 10
CXPB = 0.5
MUTPB = 0.2

def run_ga():
    pop = toolbox.population(n=POP_SIZE)
    # Evaluate the entire population
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    hof = tools.HallOfFame(1)

    for gen in range(N_GEN):
        # Selection
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Mutation
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Re-evaluate invalid individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.fitness.values = toolbox.evaluate(ind)

        pop[:] = offspring
        hof.update(pop)

    best_ind = hof[0]
    return best_ind


start_time = time.time()
best_individual = run_ga()
ga_runtime_seconds = time.time() - start_time

selected_idx_ga = [i for i, bit in enumerate(best_individual) if bit == 1]
if not selected_idx_ga:
    # Fallback: if somehow GA ended with zero features, use baseline set
    selected_features_ga = baseline_features
else:
    selected_features_ga = [feature_names[i] for i in selected_idx_ga]

# Evaluate GA solution explicitly
Xtr_ga = X_train.iloc[:, selected_idx_ga]
Xval_ga = X_val.iloc[:, selected_idx_ga]

ga_clf = RandomForestClassifier(random_state=42)
ga_clf.fit(Xtr_ga, y_train)
y_val_ga = ga_clf.predict(Xval_ga)
ga_accuracy = accuracy_score(y_val, y_val_ga)

print("GA selected features:", selected_features_ga)
print("GA accuracy:", ga_accuracy)
print("GA runtime (s):", ga_runtime_seconds)

# Save GA metrics
ga_metrics = {
    "ga_accuracy": float(ga_accuracy),
    "ga_num_features": len(selected_features_ga),
    "ga_runtime_seconds": float(ga_runtime_seconds),
    "selected_feature_names": selected_features_ga,
}
with open(args.ga_metrics, "w") as f:
    json.dump(ga_metrics, f)

# ----------------------
# SAVE FINAL SELECTED FEATURES (GA RESULT)
# ----------------------
with open(args.selected_features, "w") as f:
    json.dump({"selected_features": selected_features_ga}, f)

print("✅ Feature selection (baseline + GA) complete!")
