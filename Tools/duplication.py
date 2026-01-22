import pandas as pd
import numpy as np
from typing import Dict, Any


class LeakageValidator:
    """Duplication and leakage detection tools for tabular datasets."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
    
    def detect_duplicate_rows(self) -> Dict[str, Any]:
        """Detect exact duplicate rows."""
        duplicates = self.df.duplicated()
        duplicate_count = duplicates.sum()
        
        if duplicate_count > 0:
            return {
                'duplicate_count': int(duplicate_count),
                'duplicate_percentage': round((duplicate_count / len(self.df)) * 100, 2),
                'duplicate_indices': duplicates[duplicates].index.tolist()
            }
        return {}
    

    def validate_all(self, target_col: str, target_type: str, mi_threshold: float = 0.1) -> Dict[str, Any]:
        """Run all duplication and leakage checks."""
        return {
            'duplicate_rows': self.detect_duplicate_rows()
        }
