\# Lena-TRNN: Energy-Flow-Based Time Series Prediction



&#x20;  ```markdown

&#x20;  \[!\[PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)

&#x20;  \[!\[License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

&#x20;  ```



This repository contains the PyTorch implementation of the paper  

"Lena-TRNN: Exploring energy flow for time series prediction"

published in Neural Networks (2026, online 2025).



\## 📄 Paper Abstract (Brief)



The paper proposes a novel energy‑based architecture that eliminates the decoder by directly optimizing an energy flow through a Transformer–GRU hybrid. The model outputs an energy score for each prediction, which is shown to be strongly correlated with the prediction error. This enables confidence estimation and out‑of‑distribution detection without additional calibration.



\## 🔧 Method Overview



\- Core idea: Replace the traditional decoder with an iterative energy‑minimization process.

\- Architecture: 

&#x20; - Transformer encoder processes the input sequence.  

&#x20; - GRU updates hidden states guided by an energy function.  

&#x20; - No explicit decoder; predictions are obtained by gradually lowering the energy.

\- Key result: Energy score correlates with prediction error (Pearson correlation \~0.6 on ETTh1).



\## 📦 Requirements



\- Python 3.8+

\- PyTorch 1.10+

\- NumPy

\- Pandas

\- Matplotlib



Install all dependencies with:



```bash

pip install -r requirements.txt

```



```bash

pip freeze > requirements.txt

```



\## 🚀 Quick Start



1\. Clone this repository

&#x20;  ```bash

&#x20;  git clone https://github.com/stellachen030611-hue/Lena-TRNN

&#x20;  cd Lena-TRNN

&#x20;  ```



2\. Prepare the dataset

&#x20;  The code expects the ETTm1 dataset. Place `ETTm1.csv` in the `data/` folder (or adjust the path in `data\_utils.py`).  

&#x20;  You can download the dataset from \[ETT Dataset](https://github.com/zhouhaoyi/ETDataset).



3\. Run training and visualisation

&#x20;  ```bash

&#x20;  python train.py

&#x20;  ```

&#x20;  This will train the model and generate three output images:



Prediction Evolution – shows how iterative energy updates refine the forecast.

!\[Prediction evolution](images/prediction\_evolution.png)



Energy vs. Error – compares the energy score with the prediction error over time.

!\[Energy vs error](images/energy\_vs\_error.png)



Scatter Plot – quantifies the correlation (Pearson coefficient ≈ 0.6).

!\[Scatter energy error](images/scatter\_energy\_error.png)



\## 📂 Code Structure



| File | Description |

|------|-------------|

| `data\_utils.py` | Data loading, train/val/test split, sequence generation |

| `model.py` | Definition of the Lena‑TRNN model (Transformer + GRU + energy update) |

| `train.py` | Training loop, validation, energy evaluation, and plot generation |



\## 📊 Experimental Results



The following figures are produced after training:



\- Prediction Evolution – shows how iterative energy updates refine the forecast.

\- Energy vs. Error – compares the energy score and the prediction error over time.

\- Scatter Plot – quantifies the correlation (Pearson coefficient ≈ 0.6).



These results confirm the main claim of the paper: the energy score serves as a reliable proxy for prediction error.



\## 📚 Citation



If you use this code for your research, please cite the original paper:



```

@article{lena2026lenatrnn,

&#x20; title={Lena-TRNN: Exploring energy flow for time series prediction},

&#x20; author={...},

&#x20; journal={Neural Networks},

&#x20; year={2026}

}

```



\## 📄 License



This project is released under the MIT License. See `LICENSE` for details.



\---



Note: The code is provided for research and educational purposes. The implementation may not exactly match the original paper’s hyperparameters; feel free to adjust them in `train.py`.

```



