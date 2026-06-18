# Transformer für geometrische Objekterkennung

**Autor:** Matthes Fritsche 
**Datum:** 2026-06-10
**Universität:** Technische Universität Dresden  

---

## Inhaltsverzeichnis

1. [Zusammenfassung (Abstract)](#1-zusammenfassung-abstract)
2. [Einleitung](#2-einleitung)
3. [Verwandte Arbeiten](#3-verwandte-arbeiten)
4. [Methodik](#4-methodik)
5. [Ergebnisse](#5-ergebnisse)
6. [Diskussion](#6-diskussion)
7. [Fazit und Ausblick](#7-fazit-und-ausblick)
8. [Literaturverzeichnis](#8-literaturverzeichnis)
9. [Appendix](#9-appendix)

---

## 1. Zusammenfassung (Abstract)

### Abstract

<!-- TODO: Kurze englische Zusammenfassung (5-8 Sätze)
- Problemstellung
- Methode
- Hauptergebnisse
- Implikationen
-->

---

## 2. Einleitung

### 2.1 Motivation und Problemstellung

<!-- TODO: Warum ist dieses Problem relevant?
- Hintergrund zu Transformern und Attention-Mechanismen
- Praktische Anwendungen
-->

### 2.2 Forschungsfragen und Hypothesen

<!-- TODO: Konkrete Forschungsfragen formulieren
- Hauptfrage: Wie beeinflussen Architekturparameter die Performance?
- Unterfragen zu Heads, Layers, d_model
- Arbeitshypothesen basierend auf initialen Resultaten
-->

### 2.3 Zielsetzung und Beitrag dieser Arbeit

<!-- TODO: Was ist der konkrete Beitrag?
- Erweiterung des bisherigen Sweeps (1-4 zu 5-8)
- Systematische Evaluation
- Praktische Implikationen für Architektur-Design
-->

---

## 3. Verwandte Arbeiten

### 3.1 Grundlagen: Transformer und Attention

<!-- TODO: Basis-Konzepte
- Attention-Mechanismus (Vaswani et al. 2017)
- Self-Attention und Multi-Head Attention
- Query-Key-Value Konzept
- Positional Encoding
-->

**Relevante Paper:**
- [ ] Vaswani et al. (2017): "Attention is All You Need"
- [ ] Weitere Paper zu Attention-Mechanismen


**Relevante Papiere:**
- [ ] Dosovitsky et al. (2021): "An Image is Worth 16x16 Words"
- [ ] Bronstein et al. (2021): "Geometric Deep Learning"
- [ ] Weitere Vision Transformer Papiere

### 3.2 Architektur-Optimierung und Hyperparameter-Tuning

<!-- TODO: Wie optimiert man Netzwerk-Architektur?
- Hyperparameter-Tuning Strategien
- Trade-offs zwischen Modell-Größe und Performance
- Bisherige Findings
-->

**Relevante Papiere:**
- [ ] Baker et al. (2016) / Zoph & Quah (2016): Neural Architecture Search
- [ ] Weitere AutoML / NAS Papiere

### 3.3 Bisherige Erkenntnisse aus dem Projekt

<!-- TODO: Zusammenfassung der initialen Sweep-Resultate
- Findings aus layers/heads 1-4
- Motivation für Erweiterung zu 5-8
- Beobachtete Trends
-->

---

## 4. Methodik

### 4.1 Datensatz und Datengeneration

#### 4.1.1 Synthethisches Datenset

<!-- TODO: Beschreibung des Datensatzes
- Koordinaten-Bereich: [-0.5, 0.5]²
- Kreis-Radien: U(0.1, 0.25)
- Anzahl Punkte pro Sample: 200
- Größe des Trainingssatzes: ?
- Größe des Validationssatzes: ?
-->

#### 4.1.2 Rauschen und Clutter

<!-- TODO: Datenaugmentation und Robustness
- Rausch-Modell (std = 0.05)
- Clutter (fraction = 0.1)
- Warum diese Parameter?
-->

### 4.2 Modellarchitektur

#### 4.2.1 Transformer-Basis-Design

<!-- TODO: Architektur-Details
- Input-Embedding
- Transformer-Block Struktur
- Output-Head für Kreis-Parameter
- Zielgröße: Center (x,y) und radius?
-->

#### 4.2.2 Architektur-Parameter

<!-- TODO: Parametisiertes Design
- Untersuchte Layer-Bereiche: 5-8
- Untersuchte Head-Bereiche: 5-8
- d_model Basis: 120
- Spezialfall heads=7: d_model=126
- Begründung für diese Constraints
-->

**Tabelle: d_model Auswahl**

| Anzahl Heads | d_model | Begründung |
|--------------|---------|-----------|
| 5 | 120 | 120 % 5 = 0 ✓ |
| 6 | 120 | 120 % 6 = 0 ✓ |
| 7 | 126 | 120 % 7 ≠ 0, daher 126 (nächstes gerades Vielfaches) |
| 8 | 120 | 120 % 8 = 0 ✓ |

### 4.3 Training und Hyperparameter

<!-- TODO: Training-Setup
- Optimizer: ...
- Learning Rate: 1e-4
- Batch Size: 64
- Epochs: 15.000
- Loss Function: ...
- Early Stopping Kriterien: ...
-->

**Trainingskonfiguration:**

```
epochs: 15.000
batch_size: 64
learning_rate: 1e-4
noise: True (std=0.05)
clutter: True (fraction=0.1)
num_points: 200
```

### 4.4 Evaluations-Metriken

<!-- TODO: Wie wird Performance gemessen?
- Primäre Metrik: min_loss über letzte 100 Epochen (last_window_loss_min)
- Sekundäre Metriken: ? (MSE, MAE, Radius-Accuracy?)
- Visualisierungen: Heatmaps, Loss-Kurven, Sample-Plots
- Warum diese Metrik-Auswahl?
-->

#### 4.4.1 Primäre Metrik

- **Metric:** Minimum Loss über die letzten 100 Epochen (`last_window_loss_min`)
- **Begründung:** <!-- TODO -->

#### 4.4.2 Sekundäre Metriken und Visualisierungen

<!-- TODO: Weitere Evaluations-Dimensionen
- Convergence Speed
- Robustheit gegenüber Rauschen
- Qualitative Analyse: Sample-Predictions pro Modell
-->

### 4.5 Experimentelles Design

<!-- TODO: Ablauf des Sweeps
- Matrix: 4 × 4 Experimente (layers 5-8, heads 5-8)
- Randomisierung / Seeds
- Replikationen
- Hardware / Computational Resources
-->

---

## 5. Ergebnisse

### 5.1 Zusammenfassung der Trainingsergebnisse

<!-- TODO: Überblick über alle Runs
- Tabelle mit Resultaten für alle 16 Modell-Kombinationen
- Min Loss Werte
- Konvergenz-Verhalten
-->

**Tabelle: Experimentelle Resultate (layers × heads × d_model)**

| Layers | Heads | d_model | Min Loss (last 100) | Bemerkungen |
|--------|-------|---------|-------------------|-------------|
| 5 | 5 | 120 | <!-- TODO --> | |
| 5 | 6 | 120 | <!-- TODO --> | |
| 5 | 7 | 126 | <!-- TODO --> | *Breiteres Modell* |
| 5 | 8 | 120 | <!-- TODO --> | |
| ... | ... | ... | ... | |

### 5.2 Heatmap-Analyse

<!-- TODO: Visualisierung des Parameter-Raums
- Heatmap: Layers vs. Heads
- Trends erklärt
- Best-Performing Kombinationen
- Worst-Performing Kombinationen
- Statistische Signifikanz?
-->

**Abbildung 5.1:** <!-- Heatmap einfügen -->

### 5.3 Validierungs-Visualisierungen

<!-- TODO: Pro-Modell Analyse
- Loss-Kurven während Training
- Convergence Pattern
- 10 Sample-Predictions pro Modell
- Ground Truth vs. Predictions
-->

**Abbildung 5.2 - 5.17:** <!-- Sample-Plots für ausgewählte Modelle -->

### 5.4 Vergleich: heads=7 (d_model=126) vs. andere

<!-- TODO: Auswirkung der d_model-Anpassung
- Ist heads=7 benachteiligt/bevorteiligt durch größere Breite?
- Fair-Comparison Überlegungen
- Ablation: Wie würde heads=7 bei d_model=120 performen?
-->

---

## 6. Diskussion

### 6.1 Interpretation der Hauptergebnisse

<!-- TODO: Was bedeuten die Resultate?
- Bestätigung / Widerlegung der Hypothesen
- Trends: Mehr Heads → besser? Mehr Layers → besser?
- Nicht-lineare Effekte?
- Optimale Architektur-Region
-->

### 6.2 Vergleich mit initialen Findings (layers/heads 1-4)

<!-- TODO: Konsistenz über beide Sweeps
- Waren die Trends aus 1-4 weiterhin sichtbar in 5-8?
- Gibt es neue Erkenntnisse?
- Saturations-Effekte bei größeren Werten?
-->

### 6.3 Computationale Effizienz

<!-- TODO: Trade-off Analyse
- Modell-Größe vs. Performance
- Training Time Complexity
- Inference Speed (falls gemessen)
- Praktische Empfehlungen für Deployment
-->

### 6.4 Limitations dieser Studie

<!-- TODO: Kritische Reflexion
- Synthethischer Datensatz: Generalisierung auf reale Daten?
- Kleine Modelle: Skalieren Erkenntnisse zu größeren Transformers?
- Single Seed vs. Multiple Runs: Varianz der Resultate?
- Task-Spezifität: Geometrische Objekterkennung vs. allgemeine Vision Tasks
-->

---

## 7. Fazit und Ausblick

### 7.1 Hauptaussagen

<!-- TODO: Zusammenfassung der Kernerkenntnisse
- Was haben wir gelernt?
- Architektur-Empfehlungen
- Praktische Implikationen
-->

---

## 8. Literaturverzeichnis

### Primäre Referenzen

<!-- TODO: Alle zitierten Papiere mit vollständigen Citations
Format: Author(Year). Title. Journal/Conference. Link/DOI
-->

**Attention und Transformers:**
- [ ] Vaswani, A., et al. (2017). "Attention is All You Need." NeurIPS.
- [ ] <!-- Weitere Papiere -->

**Vision Transformers:**
- [ ] Dosovitsky, A., et al. (2021). "An Image is Worth 16x16 Words." ICLR.
- [ ] <!-- Weitere Papiere -->

**Geometrisches Deep Learning:**
- [ ] Bronstein, M. M., et al. (2021). "Geometric Deep Learning." JMLR.
- [ ] <!-- Weitere Papiere -->

**Neural Architecture Search:**
- [ ] Baker, B., et al. (2016). "Designing Neural Network Architectures using Reinforcement Learning." ICLR.
- [ ] <!-- Weitere Papiere -->

---

## 9. Appendix

### A. Zusätzliche Tabellen und Figuren

<!-- TODO: Umfangreiche Resultate, die den Haupttext zu sehr belasten würden
- Detaillierte Metriken pro Modell
- Zusätzliche Heatmaps (z.B. Training Loss, Convergence Speed)
- Parameter-Sensitivitäts-Analysen
-->

### B. Code und Reproduzierbarkeit

<!-- TODO: Verweise auf Code-Repositories, Train-Skripte
- sweep_models.py
- evaluate.py
- data_generator.py
- Kommando zum Reproduzieren (mit Parametern)
-->

**Reproduzierungs-Kommando:**

```bash
.venv/bin/python sweep_models.py \
  --layers-min 5 --layers-max 8 \
  --heads-min 5 --heads-max 8 \
  --d-model 120 --num-points 200 \
  --epochs 15000 --batch-size 64 \
  --lr 0.0001 \
  --noise --clutter \
  --run-root runs_5to8 \
  --skip-existing
```

---

**Letzte Aktualisierung:** 2026-06-10
