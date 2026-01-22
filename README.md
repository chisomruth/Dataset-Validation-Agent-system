# LOCAL-FIRST DATASET VALIDATION AGENT (NO FIXES)

This project is a **read-only agent** that inspects **tabular datasets**, identifies issues, and produces a structured report. The agent **does not modify the dataset** — it only detects and reports problems.

#### Motivation
Data is the backbone of modern AI systems, but even small inconsistencies, missing values, or duplicates can break downstream models. Cleaning datasets manually is slow and error-prone.  

This agent provides a **safe, local-first solution** to quickly assess dataset quality before any modeling or fine-tuning. It is lightweight, practical, and designed for real-world ML pipelines.

- The agent is business-agnostic.
- It can be integrated into existing pipelines or used standalone.
- It Currently focuses on tabular datasets.
- Future updates may expand capabilities into other dataset modalities.

---

### Architecture
                   ┌────────────────────────┐
                   │  Validation Agent      │
                   │  (Read-Only / Local)  │
                   └───────────┬────────────┘
                               │
      ┌───────────────┬───────────────┬───────────────┐
      │               │               │               │
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│ Schema Checks │ │ Value Checks  │ │ Duplication & │
│               │ │               │ │ Leakage Checks│
└───────────────┘ └───────────────┘ └───────────────┘
      │               │               │
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│ Column types  │ │ Missing/      │ │ Exact/Near-   │
│ & formats     │ │ placeholder   │ │ duplicate     │
│ Columns &     │ │ values        │ │ rows          │
│ constraints   │ │ Outliers      │ │ Leakage       │
└───────────────┘ │ Rare values   │ └───────────────┘
                  └───────────────┘



### Capabilities

#### Schema Checks
- Validate column types and formats  
- Detect missing columns  
- Check basic constraints (e.g., unique IDs)  

#### Value Checks
- Identify missing or placeholder values  
- Detect outliers and unusual distributions  
- Highlight rare categories  

#### Duplication Checks
- Detect duplicate rows  


> ⚠️ **Note:** This project is under active development. The agent currently only detects and reports issues; it does not modify the dataset.
