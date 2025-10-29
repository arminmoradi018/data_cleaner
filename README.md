# 🧹 Data Cleaner

An **interactive Shiny for Python application** for cleaning, analyzing, and visualizing datasets.  
Easily handle missing data, normalize or standardize features, drop unnecessary columns, and explore your dataset through analysis tables and rich visualizations — all in one place.

[![Tests](https://github.com/arminmoradi018/data_cleaner/actions/workflows/ci.yml/badge.svg)](https://github.com/arminmoradi018/data_cleaner/actions/workflows/ci.yml)

---

## 🚀 Features

- 🧼 **Data Cleaning** – Drop columns, fill missing values with (mean, median, zero, or drop rows), and apply normalization/standardization.  
- 📊 **Descriptive Analysis** – View summary statistics (mean, std, min, max, etc.) and column-level metadata (type, missing count, unique count).  
- 🎨 **Visualization Suite** – Create scatter, histogram, boxplot, and pairwise relationship plots with Seaborn and Matplotlib.  
- 🧭 **Dimensionality Reduction** – Perform **PCA** and **t-SNE** in 2D or 3D, with optional color encoding by target column.  
- 🌙 **Dark Mode Support** – Toggle light/dark UI seamlessly.  
- 💾 **Download Cleaned Data** – Export your processed dataset as CSV.  

---

## 🧠 Tech Stack

| Category        | Tools                                          |
| ---------------- | ---------------------------------------------- |
| Framework        | [Shiny for Python](https://shiny.posit.co/py/) |
| Data Processing  | Pandas, NumPy, Scikit-learn                   |
| Visualization    | Seaborn, Matplotlib                            |
| Testing          | Pytest                                         |
| CI/CD            | GitHub Actions                                 |

---
## 🌐 Data Source

You can use this dataset to test the app:
[🐧 Download penguins.csv](https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv)

---

## 🧩 Project Structure

```bash
data_cleaner/
├── src/
│   ├── __init__.py
│   ├── app.py
│   ├── server.py
│   └── ui.py
├── test/
│   ├── __init__.py
│   └── test_app.py
├── .github/workflows/ci.yml
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

---

## 🧪 Run Locally

```bash
# 1️⃣ Clone the repository
git clone https://github.com/arminmoradi018/data_cleaner.git
cd data_cleaner

# 2️⃣ Create & activate virtual environment
python -m venv .venv
.venv\Scripts\activate   # on Windows
# or
source .venv/bin/activate  # on macOS/Linux

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Run the Shiny app
shiny run --reload --launch-browser src/app.py
```
---

## 🧪 Testing

This project includes **unit tests** (Pytest) verifying core data-processing logic:

1. ✅ **File Upload** — Ensures CSV files load correctly.  
2. 🧹 **Data Cleaning** — Tests column removal, missing-value handling, normalization, and standardization.  
3. 📊 **Data Analysis** — Validates statistical summaries and metadata output.  
4. 🔍 **PCA & t-SNE** — Ensure proper dimensionality reduction and visualization of high-dimensional features.  

### 🧰 Running Tests Locally

To run all tests locally:

```bash
pytest -v

```
---

## 🤖 Continuous Integration (CI)

This project uses **GitHub Actions** for continuous integration.  
Every time you **push** a commit or open a **pull request**, all automated tests are executed in a clean environment to ensure code quality and stability.

---

### 🧱 CI Workflow Overview

The CI workflow is defined in the following file:
.github/workflows/ci.yml

It performs the following steps:

1. **Checkout the repository** – downloads the latest version of the code.
2. **Set up Python** – installs Python 3.11.
3. **Install dependencies** – installs required packages from `requirements.txt`.
4. **Run tests** – executes all Pytest tests located in the `test/` directory.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](./LICENSE) file for details.

---

## 👤 Author  
**Armin Moradi**  
🎓 AI student (3rd semester) at JKU Linz  
📘 This project was originally a smaller assignment from one of my second-semester courses in the Artificial Intelligence program, which I later expanded and enhanced on my own.  
📫 Arminmoradi018@gmail.com

