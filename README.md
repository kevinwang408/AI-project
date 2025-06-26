# AI-based Solar Radiation Prediction

This project predicts the next time point's solar radiation intensity in Hawaii using time series data and a variety of machine learning and deep learning models.

## Table of Contents

- [Project Overview](#project-overview)
- [Models Implemented](#models-implemented)
- [Data](#data)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Overview

This project explores multiple machine learning and deep learning models to forecast solar radiation intensity for the next time point using historical time series data collected in Hawaii. The primary goal is to compare model performances and identify the most accurate approach.

**graphical abstract**

<p align="center">
  <img src="https://github.com/user-attachments/assets/11c1bf63-84ab-454c-b35d-26ccade3115d" alt="graphical abstract" width="500"/>
</p>


## Models Implemented

- 1D CNN (Convolutional Neural Network)
- GRU (Gated Recurrent Unit)
- LSTM (Long Short-Term Memory)
- MLP (Multi-Layer Perceptron)
- Transformer
- Regression Tree
- Random Forest Regression
- SVR (Support Vector Regression)
- TCN (Temporal Convolutional Network)

## Data

The dataset consists of time series records of solar radiation intensity measured in Hawaii.

- **Source:** [Describe your data source or add a link if public]
- **Format:** [Briefly describe data columns, file format, etc.]

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/kevinwang408/AI-project.git
    cd AI-project
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *(List any additional installation notes here, e.g., Python version, CUDA requirements, etc.)*

## Usage

All models are organized in a modular fashion under the `model modularization` directory. Each subfolder within this directory corresponds to a specific model:

```
model modularization/
  ├── 1D_CNN/
  ├── GRU/
  ├── LSTM/
  ├── MLP/
  ├── Transformer/
  ├── Regression_Tree/
  ├── Random_Forest_Regression/
  ├── SVR/
  └── TCN/
```

Each model folder contains a `main.py` file. To use or train a specific model, navigate to the corresponding folder and execute `main.py`:

```bash
cd "model modularization/ModelName"
python main.py
```
Replace `ModelName` with the name of the model you wish to run (e.g., `LSTM`, `GRU`, etc.).

**Important Notes:**
- **Parameter Modification:**  
  If you wish to adjust the model’s parameters (such as learning rate, number of epochs, etc.), you will need to directly modify the relevant code inside `main.py` or other source files within the specific model folder.
- **Data Requirements:**  
  Make sure your data is correctly formatted and placed as expected by each model’s code.
- **Environment:**  
  Ensure that all required dependencies (see [Installation](#installation)) are installed and your Python environment is properly set up.

Feel free to explore and modify the models as needed for your experiments!

## Results

| Model       | MAE   | RMSE  | R²    |
|-------------|-------|-------|-------|
| 1D CNN      |   54.32    |   103.79    |   0.9    |
| GRU         |  40.68     |   90.94    |    0.93   |
| LSTM        |   46.04    |   90.47    |   0.92    |
| MLP         |   45.39    |   91.4    |    0.92   |
| Transformer |    32.18   |    77.96   |    0.94   |
| Regression Tree |  42.7 |   96.09    |   0.92    |
| Random Forest |  54.72   |   106.24    |    0.9   |
| SVR         |   46.53    |    91.94   |   0.92    |
| TCN         |   50.55    |    93.89   |    0.92   |

Results show that Transformer model have the best performance on solar radiation prediction

## Contributing

Contributions are welcome! Please open an issue or pull request for improvements or new features.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more information.

## Contact

Maintained by [Kevin Wang](mailto:wang858107473@gmail.com).
