# Probabilistic-U-Net-Evaluation-Wue_Lab

Dies ist das ergänzende Repository zur Seminararbeit "Bayesian Deep Learning-Methoden für Semantic Segmentation bei biomedizinischen Anwendungen" am Lehrstuhl von Prof. Dr. Christoph Flath, Julius-Maximilians-Universität Würzburg.

Zur Reproduktion der Ergebnisse wird die Verwendung von Google Drive zum Entpacken und Speichern der Daten sowie Google Colaboratory für die Ausführung des Codes empfohlen.

## Quick Start

- Lege in der Google Drive ein Google Colaboratory-Notebook an (Neu --> Mehr --> Google Colaboratory)

- Verbinde Notebook mit Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

- Wechsle Verzeichnis zu Google Drive:
```python
cd 'drive/My Drive'
```

- Kopiere GitHub-Repository in Google Drive:
```python
! git clone https://github.com/FelixBey/Probabilistic-U-Net-Evaluation-Wue_Lab.git
```

Nach dem Ausführen dieser Schritte sollten die Notebooks mit Daten im Zielordner 'Probabilistic-U-Net-Evaluation-Wue_Lab' abliegen.

Lade im nächsten Schritt die [vortrainierten Gewichte](https://drive.google.com/drive/folders/1heyrzuPJxlgQPXrS1bOHIbq5G4bqBHVp?usp=sharing) für die Evaluation herunter.

Lade die vortrainierten Gewichte in den Google Drive-Ordner 'Probabilistic-U-Net-Evaluation-Wue_Lab-->model-->pretrained_weights'. Falls der Ordner 'pretrained_weights' noch nicht existiert, erstelle ihn zuerst.

Die Evaluationen der Arbeit können nun repliziert werden:

- Führe zuerst das
[Notebook für das Erstellen der Samples und qualitativen Evaluationen](https://github.com/FelixBey/Probabilistic-U-Net-Evaluation-Wue_Lab/blob/main/wue_lab_evaluation_plots.ipynb) aus
aus
- Anschließend kann auch das [Notebook für das Erstellen der quantitativen Evaluationen](https://github.com/FelixBey/Probabilistic-U-Net-Evaluation-Wue_Lab/blob/main/wue_lab_evaluation_plots.ipynb) ausgeführt werden
