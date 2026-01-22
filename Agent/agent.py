import pandas as pd
from langgraph.graph import StateGraph, END
from typing import Dict, Any, TypedDict
from Tools.schema import SchemaValidator
from Tools.value import ValueValidator
from Tools.duplication import LeakageValidator
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()

class ValidationState(TypedDict):
    df: pd.DataFrame
    target_column: str
    target_type: str
    schema_report: Dict[str, Any]
    value_report: Dict[str, Any]
    leakage_report: Dict[str, Any]
    llm_analysis: str
    final_report: Dict[str, Any]

class DatasetValidationAgent:
    """LangGraph-based dataset validation agent with LLM analysis."""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the validation workflow graph."""
        workflow = StateGraph(ValidationState)
        
        workflow.add_node("schema_validation", self._schema_validation)
        workflow.add_node("value_validation", self._value_validation)
        workflow.add_node("leakage_validation", self._leakage_validation)
        workflow.add_node("llm_analysis", self._llm_analysis)
        workflow.add_node("generate_report", self._generate_report)
        
        workflow.set_entry_point("schema_validation")
        workflow.add_edge("schema_validation", "value_validation")
        workflow.add_edge("value_validation", "leakage_validation")
        workflow.add_edge("leakage_validation", "llm_analysis")
        workflow.add_edge("llm_analysis", "generate_report")
        workflow.add_edge("generate_report", END)
        
        return workflow.compile()
    
    def _schema_validation(self, state: ValidationState) -> ValidationState:
        """Run schema validation checks."""
        validator = SchemaValidator(state["df"], target_col=state["target_column"])
        state["schema_report"] = validator.validate_all()
        return state
    
    def _value_validation(self, state: ValidationState) -> ValidationState:
        """Run value quality checks."""
        validator = ValueValidator(state["df"], target_col=state["target_column"])
        state["value_report"] = validator.validate_all()
        return state
    
    def _leakage_validation(self, state: ValidationState) -> ValidationState:
        """Run duplication and leakage checks."""
        validator = LeakageValidator(state["df"])
        state["leakage_report"] = validator.validate_all(
            target_col=state["target_column"],
            target_type=state["target_type"]
        )
        return state
    
    def _llm_analysis(self, state: ValidationState) -> ValidationState:
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
    
    def _generate_report(self, state: ValidationState) -> ValidationState:
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
    
    def validate_dataset(self, df: pd.DataFrame, target_column: str, target_type: str) -> Dict[str, Any]:
        """Run complete dataset validation."""
        initial_state = ValidationState(
            df=df,
            target_column=target_column,
            target_type=target_type,
            schema_report={},
            value_report={},
            leakage_report={},
            llm_analysis="",
            final_report={}
        )
        
        result = self.graph.invoke(initial_state)
        return result["final_report"]
