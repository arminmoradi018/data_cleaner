import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA


# -------------------------------
# Fixtures
# -------------------------------
@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing"""
    return pd.DataFrame({
        "A": [1, 2, np.nan, 4],
        "B": [10, 20, 30, 40],
        "C": ["x", "y", "z", "x"]
    })


# -------------------------------
# File upload tests
# -------------------------------
def test_upload_csv(tmp_path, sample_df):
    """Check that CSV file loads correctly"""
    test_file = tmp_path / "data.csv"
    sample_df.to_csv(test_file, index=False)

    df_loaded = pd.read_csv(test_file)
    assert not df_loaded.empty
    assert list(df_loaded.columns) == ["A", "B", "C"]
    assert df_loaded.shape == (4, 3)


# -------------------------------
# Data cleaning tests
# -------------------------------
def test_drop_columns(sample_df):
    """Test column removal"""
    df = sample_df.copy()
    cols_to_drop = ["C"]
    df_clean = df.drop(columns=cols_to_drop)
    assert "C" not in df_clean.columns
    assert df_clean.shape[1] == 2


def test_fillna_mean(sample_df):
    """Test replacing NaN values with mean"""
    df = sample_df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    assert not df["A"].isna().any()
    assert np.isclose(df["A"].iloc[2], df["A"].mean(), atol=1e-6)


def test_normalize_columns(sample_df):
    """Test normalization (0-1 scaling)"""
    df = sample_df.copy()
    scaler = MinMaxScaler()
    df[["A", "B"]] = scaler.fit_transform(df[["A", "B"]])
    assert np.isclose(df["A"].min(), 0)
    assert np.isclose(df["A"].max(), 1)
    assert np.isclose(df["B"].max(), 1)


def test_standardize_columns(sample_df):
    """Test standardization (0 mean, 1 std)"""
    df = sample_df.copy()
    df = df.fillna(df.mean(numeric_only=True))  # Fill NaN before scaling
    scaler = StandardScaler()
    df[["A", "B"]] = scaler.fit_transform(df[["A", "B"]])
    assert np.isclose(df["A"].mean(), 0, atol=1e-6)
    assert np.isclose(df["A"].std(ddof=0), 1, atol=1e-6)  # ✅ fixed here




# -------------------------------
# Data analysis tests
# -------------------------------
def test_analysis_dataframe(sample_df):
    """Check data analysis output structure"""
    df = sample_df.copy()
    analysis = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isna().sum(),
        "Unique Values": df.nunique()
    })
    assert "Column" in analysis.columns
    assert analysis["Missing Values"].sum() == 1


# -------------------------------
# PCA dimensionality reduction test
# -------------------------------
def test_pca_reduction(sample_df):
    """Ensure PCA runs correctly with numeric columns"""
    df = sample_df.select_dtypes(include=np.number).dropna()
    X = StandardScaler().fit_transform(df)
    pca = PCA(n_components=2)
    transformed = pca.fit_transform(X)
    assert transformed.shape[1] == 2
    assert transformed.shape[0] == len(df)
