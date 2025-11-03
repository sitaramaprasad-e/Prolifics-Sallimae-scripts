# PCPT Implementation Guide

This README describes how to use **PCPT (Prolifics Code Profiler and Transformer)** to generate documentation and business rule artifacts.

> Note: Run all commands from the root of the `sally-mae-example` directory. Outputs will appear in `docs`. There are two sets of code 1) stored procedure 2) Salesforce code

---

## 1. Analyze: Extract Code Rules from Stored Proc

The code lives in code/stored_proc.

### 1.1 Prepare PCPT

Copy the hints and prompts files:

```bash
cp hints/* ~/.pcpt/hints
cp prompts/* ~/.pcpt/prompts
cp filters/* ~/.pcpt/filters
mkdir -p ~/.model
cp .model/rule_categories.json ~/.model/rule_categories.json
```

Reset all data:

**(Optional) Clean Previous Outputs**
```bash
rm -rf ~/.model/artifacts.json ~/.model/business_rules.json ~/.model/code_business_rules.json ~/.model/documented_business_rules.json ~/.model/executions.json ~/.model/feedback.json ~/.model/runs.json ~/.model/sources.json
rm -rf ./docs/sf/* ./docs/stored_proc/* ./docs/categorise-rule/*
```

### 1.2 Generate Reports

Use the following commands to generate domain models, use cases, sequence diagrams, and business logic reports for stored procedure code.

> Note: Only the domain model and business logic reports are required for rule extraction and transformation. The use cases and sequence diagrams are optional.

**Generate Domain Model**
```bash
pcpt.sh domain-model --output docs/stored_proc --domain-hints stored_proc.hints --visualize code/stored_proc
```

**Generate Use Cases** (requires domain model)
```bash
pcpt.sh use-cases --output docs/stored_proc --domain docs/stored_proc/domain_model_report/domain_model_report.txt --domain-hints stored_proc.hints --visualize code/stored_proc
```

**Generate Sequence Diagram**
```bash
pcpt.sh sequence --output docs/stored_proc --domain-hints stored_proc.hints --visualize code/stored_proc
```

**Extract Business Logic** (requires domain model)
```bash
pcpt.sh business-logic --output docs/stored_proc --domain docs/stored_proc/domain_model_report/domain_model_report.txt --domain-hints stored_proc.hints code/stored_proc
```

### 1.3 Ingest Code Rules from Generated Reports

This step extracts structured business rules from the generated documentation for later correlation and transformation.

**Initialize Python Environment**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Ingest Code Rules from Report and Add to Model**

```bash
python tools/ingest_rules.py docs/stored_proc/business_logic_report/business_logic_report.md
```

---

## 2. Correlate: Match Code Rules with Document Rules (Optional)

This step extracts business rules from documentation and matches them with code-extracted rules to update those code-extracted rules with extra details from the documentation. Only those extracted rules that can be correlated with documentation rules will be updated.

**Extract Rules from Document**

```bash
python tools/extract_rules_from_doc.py
```

**Correlate Document Rules with Code Rules**

```bash
PYTHONWARNINGS="ignore" python tools/correlate_code_and_doc.py
```

---

## 3. Organize Rules

### 3.1 Categorize Rules

**Categorize Rule** (requires rules)
```bash
python tools/categorise_rules.py
```
### 3.2 Suggest Models

**Suggest Models** (requires rules)
```bash
python tools/suggest_models.py
```
Then choose from one of the prompts listed. Each produces a different report named after the prompt.

IMPORTANT: First make sure you copy all the prompts using the command in "1.1 Prepare PCPT"

---

## 4. Transform to Executable Rules (Optional)

### 4.1 Transform to Executable Rules (New Approach)

**Generate DMN Rules** (requires business logic extracted)
```bash
pcpt.sh run-custom-prompt --output docs/stored_proc docs/stored_proc/business_logic_report/business_logic_report.md generate-dmn-xml.templ
```

### 4.2 Transform to Executable Rules (Old Approach for reference)

This approach was shown as a proof of concept but has been superseded by the new approach above. Note this may no longer be compatible with the updated rule prompts/tools.

**Generate DMN Rules** (requires business logic extracted)
```bash
pcpt.sh run-custom-prompt --output docs/stored_proc docs/stored_proc/business_logic_report/business_logic_report.md generate-dmn-rules.templ
```

**Generate Drools Rules Implementation Model**
```bash
python tools/generate_drools_code.py
```

**Write Drools Rules to Code**
```bash
python tools/write_drools_code.py
```

---

## 5. Mark up a sequence diagram with rule references

```bash
python tools/markup_sequence.py code/stored_proc docs/stored_proc --domain-hints stored_proc.hints
```

---

## 6. (Optional) Steps 1-4 for Salesforce Code

You can also apply the same techniques to the Salesforce codebase.

> Outputs will appear in `docs/sf`.

### 6.1 Prepare

```bash
cp hints/* ~/.pcpt/hints
```

### 6.2 Generate Reports

**Domain Model**
```bash
pcpt.sh domain-model --output docs/sf --domain-hints sf.hints --visualize --filter sf-rules.filter code/sf
```

**Use Cases** (requires domain model)
```bash
pcpt.sh use-cases --output docs/sf --domain docs/sf/domain_model_report/domain_model_report.txt --domain-hints sf.hints --visualize --filter sf-rules.filter code/sf
```

**Sequence Diagram**
```bash
pcpt.sh sequence --output docs/sf --domain-hints sf.hints --visualize code/sf
```

**Business Logic** (requires domain model)
```bash
pcpt.sh business-logic --output docs/sf --domain docs/sf/domain_model_report/domain_model_report.txt --domain-hints sf.hints code/sf
```

## 6.3 Ingest Code Rules from Generated Reports

This step extracts structured business rules from the generated documentation for later correlation and transformation.

**Initialize Python Environment**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Ingest Code Rules from Report and Add to Model**

```bash
python tools/ingest_rules.py docs/sf/business_logic_report/business_logic_report.md
```

## 6.4 Transform to DMN

**Generate DMN Rules** (requires business logic extracted)
```bash
pcpt.sh run-custom-prompt --output docs/sf docs/sf/business_logic_report/business_logic_report.md generate-dmn-xml.templ
```

## 6.5 Mark up a sequence diagram with rule references (Automated, Use This)

```bash
python tools/markup_sequence.py code/sf docs/sf --domain-hints sf.hints
```

## 6.6 Mark up a sequence diagram with rule references (Manually, Kept for Reference Only)

**Create standard Sequence Diagram**
```bash
pcpt.sh sequence --output docs/sf --domain-hints sf.hints --visualize code/sf
```

**Copy existing sequence report to temp folder**
```bash
mkdir -p .tmp/rules-for-markup
cp docs/sf/sequence_report/sequence_report.txt
```

**Export list of current business rules for code**
```bash
python tools/export_rules_for_markup.py code/sf
```

**Markup sequence description**
```bash
pcpt.sh run-custom-prompt --input-file .tmp/rules-for-markup/sequence_report.txt --input-file2 .tmp/rules-for-markup/exported-rules.json --output docs/sf code/sf markup-sequence.templ
```

**Regenerate sequence diagram**
```bash
pcpt.sh sequence --output docs/sf --visualize docs/sf/markup-sequence/markup-sequence.md
```

---

## 7. (Optional) Steps 1-4 for Image

You can also apply the same techniques to business rules extracted from an image e.g. a workflow diagram.

> Outputs will appear in `docs/image`.

### 7.1 Prepare

```bash
cp hints/* ~/.pcpt/hints
```

### 7.2 Generate Reports

**Domain Model**
```bash
Choose one of the following two options:

pcpt.sh domain-model --image --output docs/image --domain-hints workflow_image.hints --visualize code/image
pcpt.sh domain-model --image --output docs/image --domain-hints workflow_image.hints --visualize code/image/workflow.png
```

**Business Logic** (requires domain model)
```bash
Choose one of the following two options:

pcpt.sh business-logic --image --output docs/image --domain docs/image/domain_model_report/domain_model_report.txt --domain-hints workflow_image.hints code/image
pcpt.sh business-logic --image --output docs/image --domain docs/image/domain_model_report/domain_model_report.txt --domain-hints workflow_image.hints code/image/workflow.png
```

## 7.3 Ingest Code Rules from Generated Reports

This step extracts structured business rules from the generated documentation for later correlation and transformation.

**Initialize Python Environment**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Ingest Code Rules from Report and Add to Model**

```bash
python tools/ingest_rules.py docs/image/business_logic_report/business_logic_report.md
```

## 7.4 Transform to DMN

**Generate DMN Rules** (requires business logic extracted)
```bash
pcpt.sh run-custom-prompt --output docs/image docs/image/business_logic_report/business_logic_report.md generate-dmn-xml.templ
```

---

## 8. Other Tools (Optional)

**Patching Tool**
```bash

Create new folder structure and move both code and report files into that structure.

Populate the tools/spec/patch_logs_paths.json file with the structure.

python tools/patch_logs_paths.py 

tools/ingest_rules.py ../sm-dir-layout-greg/docs/sf/force-app/main/default/business_logic_report/business_logic_report.md --force --all-runs

python tools/ingest_rules.py ../sm-dir-layout-greg/docs/fdrit-0103/business_logic_report/business_logic_report.md --force --all-runs
```

**Restore Backup Taken By Patching Tool (If Needed)**
```bash
python tools/restore_backup.py
```

**Produce Code Coverage Report**
```bash
python tools/code_coverage_report.py
```

**Generate Reports**
```bash
python tools/generate_detailed_report.py
python tools/generate_simple_report.py
```

**Code Analysis (External to This Repo)**
```bash
python ../pcpt/tools/code_analysis.py
```

**Log File Viewer (External to This Repo)**
```bash
python ../pcpt/tools/log_file_viewer.py
```

**Survey Folders (External to This Repo)**
```bash
python ../pcpt/tools/survey_folders.py
```


