import pandas as pd
import numpy as np
from typing import Dict, Any

class SchemaValidator:
    """Schema validation tools for tabular datasets."""
    
    def __init__(self, df: pd.DataFrame, target_col: str = None):
        self.df = df
        self.target_col = target_col
    
    def validate_column_types(self) -> Dict[str, Any]:
        """Validate column types and formats."""
        issues = {}
        
        for col in self.df.columns:
            # Skip target column
            if col == self.target_col:
                continue
                
            series = self.df[col]
            
            # Check object columns for mixed types
            if series.dtype == 'object':
                non_null = series.dropna()
                if non_null.empty:
                    continue
                
                # Check if mixed numeric and string
                numeric_mask = pd.to_numeric(non_null, errors='coerce').notna()
                if numeric_mask.any() and not numeric_mask.all():
                    issues[col] = {
                        'issue': 'mixed_types',
                        'counts': {
                            'numeric': int(numeric_mask.sum()),
                            'string': int((~numeric_mask).sum())
                        }
                    }
        
        return issues
    
    def detect_missing_columns(self, expected_columns: list = None) -> Dict[str, Any]:
        """Detect missing columns."""
        if not expected_columns:
            return {}
        
        current_cols = set(self.df.columns)
        expected_cols = set(expected_columns)
        missing = expected_cols - current_cols
        
        return {'missing_columns': list(missing)} if missing else {}
    
    
    def validate_all(self) -> Dict[str, Any]:
        """Run all schema validation checks."""
        return {
            'column_types': self.validate_column_types()
        }
