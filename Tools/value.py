import pandas as pd
import numpy as np
from typing import Dict, Any

class ValueValidator:
    """Value quality validation tools for tabular datasets."""
    
    def __init__(self, df: pd.DataFrame, target_col: str = None):
        self.df = df
        self.target_col = target_col
    
    def detect_missing_values(self) -> Dict[str, Any]:
        """Identify missing or placeholder values."""
        missing_info = {}
        
        for col in self.df.columns:
            # Skip target column
            if col == self.target_col:
                continue
                
            series = self.df[col]
            
            # Standard missing values
            null_count = series.isnull().sum()
            
            # Placeholder values
            placeholder_count = 0
    
            
            # Check string placeholders (case-insensitive, stripped)
            if series.dtype == 'object':
                string_placeholders = ['unknown', 'n/a', 'na', 'null', 'none', 'missing', '']
                series_lower = series.astype(str).str.strip().str.lower()
                for placeholder in string_placeholders:
                    placeholder_count += (series_lower == placeholder).sum()
            
            total_missing = null_count + placeholder_count
            
            if total_missing > 0:
                missing_info[col] = {
                    'null_values': int(null_count),
                    'placeholder_values': int(placeholder_count),
                    'total_missing': int(total_missing),
                    'missing_percentage': round((total_missing / len(series)) * 100, 2)
                }
        
        return missing_info
    
    def detect_outliers(self) -> Dict[str, Any]:
        """Detect extreme numeric outliers using IQR method."""
        outliers = {}
        
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        # Exclude target column if it's numeric
        if self.target_col and self.target_col in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=[self.target_col])
        
        for col in numeric_df.columns:
            series = numeric_df[col].dropna()
            if len(series) < 30:
                continue
            
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            # Skip if IQR is 0 (no variance)
            if IQR == 0:
                continue
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (series < lower_bound) | (series > upper_bound)
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                outliers[col] = {
                    'count': int(outlier_count),
                    'percentage': round((outlier_count / len(series)) * 100, 2),
                    'bounds': {'lower': float(lower_bound), 'upper': float(upper_bound)},
                    'extreme_values': series[outlier_mask].head(5).tolist()
                }
        
        return outliers
    
    
    def validate_all(self) -> Dict[str, Any]:
        """Run all value validation checks."""
        return {
            'missing_values': self.detect_missing_values(),
            'outliers': self.detect_outliers()
        }