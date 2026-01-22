from typing import Dict, Any
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
import os


class ValidationEngine:
    """Engine for LLM analysis and report generation."""
    
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm
    
    def llm_analysis(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to analyze validation results and provide insights."""
        
        prompt = f"""You are a data quality expert. Analyze the following dataset validation results and provide:
        1. Summary of critical issues
        2. Potential impact on ML models
        3. Recommended next steps

        Dataset: {len(state['df'])} rows, {len(state['df'].columns)} columns
        Target: {state['target_column']} ({state['target_type']})

        Validation Results:
        - Schema Issues: {state['schema_report']}
        - Value Issues: {state['value_report']}
        - Duplication/Leakage: {state['leakage_report']}

        Provide a concise analysis (max 300 words)."""
        
        messages = [
            SystemMessage(content="You are a data quality expert analyzing dataset validation results."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        state["llm_analysis"] = response.content
        return state
    
    def generate_report(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final validation report."""
        state["final_report"] = {
            "dataset_info": {
                "rows": len(state["df"]),
                "columns": len(state["df"].columns),
                "target_column": state["target_column"],
                "target_type": state["target_type"]
            },
            "validation_results": {
                "schema_validation": state["schema_report"],
                "value_validation": state["value_report"],
                "duplication_leakage": state["leakage_report"]
            },
            "ai_analysis": state["llm_analysis"]
        }
        return state
