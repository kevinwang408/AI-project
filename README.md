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

To train and evaluate models, use the provided scripts:

```bash
python train.py --model [MODEL_NAME] --data [DATA_PATH] --epochs [N]
```

**Supported model names:** `cnn1d`, `gru`, `lstm`, `mlp`, `transformer`, `regtree`, `rf`, `svr`, `tcn`

**Example:**
```bash
python train.py --model lstm --data data/hawaii_solar.csv --epochs 50
```

*(Add more usage examples, inference scripts, or Jupyter notebook instructions as needed.)*

## Results

| Model       | MAE   | RMSE  | R²    |
|-------------|-------|-------|-------|
| 1D CNN      |       |       |       |
| GRU         |       |       |       |
| LSTM        |       |       |       |
| MLP         |       |       |       |
| Transformer |       |       |       |
| Regression Tree |   |       |       |
| Random Forest |     |       |       |
| SVR         |       |       |       |
| TCN         |       |       |       |

*(Fill in your experimental results here.)*

## Contributing

Contributions are welcome! Please open an issue or pull request for improvements or new features.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more information.

## Contact

Maintained by [Kevin Wang](mailto:your.email@example.com).