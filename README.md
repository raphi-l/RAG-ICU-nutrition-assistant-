# RAG-Powered ICU Nutrition Assistant


## Project Overview

In this project, we build a complete **Retrieval-Augmented Generation (RAG) system** trained on ASPEN's 2016 and 2021 ICU Nutrition guidlines. This tool is intended to assist dietetic interns and new trainees in answering questions on establised best practice in the ICU.

## Part 1: Setup and Configuration

Installing and importing the required libraries.

<details><summary><code># Import all required libraries</code></summary>

```python
# Import all required libraries
import pymupdf4llm
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import torch
import re
from typing import List, Dict, Optional
import json

import os

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print("\n All imports successful!")
```

</details>

```text
PyTorch version: 2.14.1+cu130
CUDA available: True

 All imports successful!
```

---

## Part 2: Project Declaration

<details><summary><code>PROJECT_INFO = </code></summary>

```python

PROJECT_INFO = {
    "scenario": "Personal Interest",
    "topic": "Nutrition Intervention in the ICU",
    "description": "This RAG LLM is trained with ASPENs 2016 and 2021 practice guidelines for nutrition intervention in the ICU.",
    "target_users": "RDs and Dietetic Interns",
}

# Print your declaration
print(" PROJECT DECLARATION")
print("=" * 50)
for key, value in PROJECT_INFO.items():
    print(f"{key.replace('_', ' ').title()}: {value}")
```

</details>

```text
 PROJECT DECLARATION
==================================================
Scenario: Personal Interest
Topic: Nutrition Intervention in the ICU
Description: This RAG LLM is trained with ASPENs 2016 and 2021 practice guidelines for nutrition intervention in the ICU.
Target Users: RDs and Dietetic Interns
```

---

## Part 3: Creatinig the Knowledge Base

This is the foundation of your RAG system. Create comprehensive, well-organized documents for your chosen scenario.

## Helper Functions

<details><summary><code>import os</code></summary>

```python
import os
import zipfile

from pathlib import Path

import requests

# Setup path to data folder
data_path = Path("paper_markdowns/")

# If the image folder doesn't exist, download it and prepare it...
if data_path.is_dir():
    print(f"{data_path} directory exists.")
else:
    print(f"Did not find {data_path} directory, creating one...")
    data_path.mkdir(parents=True, exist_ok=True)

import requests

with open('paper_markdowns/ASPEN_2016_guidelines.md', 'wb') as f:
    url = 'https://raw.githubusercontent.com/raphi-l/RAG-ICU-nutrition-assistant-/refs/heads/main/paper-markdowns/ASPEN_2016_ICU_Guidelines.md'
    response = requests.get(url)
    f.write(response.content)
```

</details>

```text
Did not find paper_markdowns directory, creating one...
```

<details><summary><code>CACHE_DIR = &#x27;/content/paper_markdowns/ASPEN_2016_guidelines.md&#x27;</code></summary>

```python
CACHE_DIR = '/content/paper_markdowns/ASPEN_2016_guidelines.md'

def pdf_to_markdown(path: str) -> str:
  """
  Renders PDF as a markdown file and stores in globally defined `CACHE_DIR`
  """
  filename = os.path.splitext(os.path.basename(path))[0]
  cache_path = os.path.join(CACHE_DIR, f"{filename}.md")

  print(f"[INFO] Parsing '{filename}.pdf'...")

  markdown = pymupdf4llm.to_markdown(path)
  with open(cache_path, "w", encoding="utf-8") as f:
      f.write(markdown)
  print(f"[INFO] PDF saved as markdown to '{cache_path}'")

  return None
```

</details>

<details><summary><code>def markdown_to_dict(markdown: str,</code></summary>

```python
def markdown_to_dict(markdown: str,
                     title: str = "") -> dict:
    """Convert markdown into a dictionary of {section_header: content}"""
    sections = {}

    # find all header positions
    matches = list(SECTION_PATTERN.finditer(markdown))

    for i, match in enumerate(matches):
        header = match.group().strip()
        start = match.end()
        # clean up markdown formatting from header for use as key
        key = re.sub(r'[#*_]', '', header).strip()

        # content goes from end of this header to start of next header
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        content = f'[{title}][{key}] {markdown[start:end].strip()}'


        sections[key] = content

    return sections
```

</details>

<details><summary><code>import re</code></summary>

```python
import re

SECTION_PATTERN = re.compile(
    r'^#{2,3}\s+'
    r'(?:'
    r'\*\*[A-Z]\.\s+.+|'              # **A. through **R.
    r'_Trauma_|'
    r'_Traumatic.+|'
    r'_Open\s+Abdomen_|'
    r'_Burns_'
    r')',                              # Introduction removed
    re.IGNORECASE | re.MULTILINE
)

# usage
with open("/content/paper_markdowns/ASPEN_2016_guidelines.md", "r") as f:
    markdown = f.read()

knowledge_base = markdown_to_dict(markdown, title='ASPEN_2016_guidelines')
```

</details>

<details><summary><code># find all matches</code></summary>

```python
# find all matches
matches = SECTION_PATTERN.findall(markdown)

# print them
for match in matches:
    print(match)
```

</details>

```text
## **A. Nutrition Assessment** 
## **B. Initiate EN** 
## **C. Dosing of EN** 
## **D. Monitoring Tolerance and Adequacy of EN** 
## **E. Selection of Appropriate Enteral Formulation** 
## **F. Adjunctive Therapy** 
## **G. When to Use PN** 
## **H. When Indicated, Maximize Efficacy of PN** 
## **I. Pulmonary Failure** 
## **J. Renal Failure** 
## **K. Hepatic Failure** 
## **L. Acute Pancreatitis** 
## **M. Surgical Subsets** 
## _Trauma_
## _Traumatic Brain Injury_ 
## _Open Abdomen_
## _Burns_
## **N. Sepsis** 
## **O. Postoperative Major Surgery (SICU Admission Expected)** 
## **P. Chronically Critically Ill** 
## **Q. Obesity in Critical Illness** 
## **R. Nutrition Therapy End-of-Life Situations**
```

<details><summary><code>def clean_section(text: str) -&gt; str:</code></summary>

```python
def clean_section(text: str) -> str:

    patterns = [
        # ── 1. fix italic P values
        (r'_P_', 'P'),

        # ── 2. Question/Rationale labels
        (r'_\*\*Rationale:\*\*_', 'Rationale:'),
        (r'_\*\*Rationale:\*\*', 'Rationale:'),
        (r'\*\*Rationale:\*\*_', 'Rationale:'),
        (r'\*\*Rationale:\*\*', 'Rationale:'),
        (r'_Rationale:_', 'Rationale:'),
        (r'Rationale:_', 'Rationale:'),
        (r'_Question:_', 'Question:'),
        (r'Question:_', 'Question:'),
        (r'_Question:', 'Question:'),

        # ── 2.5 preserve mid-recommendation continuations before anything removes them
        (r'##\s+\*\*(intake\b[^\n]*)\*\*', r'\1'),
        (r'##\s+\*\*(the critically\b[^\n]*)\*\*', r'\1'),
        (r'##\s+\*\*(specialty\b[^\n]*)\*\*', r'\1'),
        (r'##\s+\*\*(in the absence\b[^\n]*)\*\*', r'\1'),

        # ── 3. fix mid-sentence page breaks
        (r'(=|\(|\)|,|\w|\*)\s*\n\n_McClave et al_\s*\n\n\d+\s*\n\n\*\*==>.*?<==\*\*\n\n\*\*Figure\s+\d+\.\*\*[^\n]*\n\n', r'\1 '),
        (r'(=|\(|\)|,|\w|\*)\s*\n\n_McClave et al_\s*\n\n\d+\s*\n\n(?:\*\*==>.*?<==\*\*\n\n\*\*Figure\s+\d+\.\*\*[^\n]*\n\n)+', r'\1 '),
        (r'(=|\(|\)|,|\w|\*)\s*\n\n_McClave et al_\s*\n\n\d+\s*\n\n\*\*Table\s+\d+\.\*\*[^\n]*\n', r'\1 '),
        (r'(=|\(|\)|,|\w|\*)\s*\n\n_McClave et al_\s*\n\n\d+\s*\n\n(?=\*\*[a-z])', r'\1 '),
        (r'(=|\(|\)|,|\w|\*)\s*\n\n_Journal of Parenteral and Enteral Nutrition[^\n]*_\s*\n\n\d+\s*\n\n\*\*==>.*?<==\*\*\n\n\*\*Figure\s+\d+\.\*\*[^\n]*\n\n', r'\1 '),
        (r'(=|\(|\)|,|\w|\*)\s*\n\n_Journal of Parenteral and Enteral Nutrition[^\n]*_\s*\n\n\d+\s*\n\n(?:\*\*==>.*?<==\*\*\n\n\*\*Figure\s+\d+\.\*\*[^\n]*\n\n)+', r'\1 '),
        (r'(=|\(|\)|,|\w|\*)\s*\n\n_Journal of Parenteral and Enteral Nutrition[^\n]*_\s*\n\n\d+\s*\n\n\*\*Table\s+\d+\.\*\*[^\n]*\n', r'\1 '),
        (r'(=|\(|\)|,|\w|\*)\s*\n\n_Journal of Parenteral and Enteral Nutrition[^\n]*_\s*\n\n\d+\s*\n\n(?=\*\*[a-z])', r'\1 '),
        (r'\n\n\d{3}\s*\n\n(?:\*\*==>.*?<==\*\*\n\n\*\*Figure\s+\d+\.\*\*[^\n]*\n\n)+', ' '),

        # ── 4. remove figures and captions
        (r'\*\*==>.*?<==\*\*\n\n\*\*Figure\s+\d+\.\*\*[^\n]*\n', ''),
        (r'\*\*==>.*?<==\*\*', ''),
        (r'==>.*?<==', ''),
        (r'\n\*\*Figure\s+\d+\.\*\*[^\n]*\n', '\n'),

        # ── 5. remove table captions then table rows
        (r'\*\*Table\s+\d+\.\*\*[^\n]*\n', ''),
        (r'(?:\|.*\n?)+', ''),

        # ── 6. remove bundle statements
        (r'-\s*[●\x07]\s*.*\n', ''),
        (r'-\s*●\s*.*\n', ''),

        # ── 7. remove journal metadata and page numbers
        (r'_Journal of Parenteral and Enteral Nutrition[^\n]*_\s*\n', ''),
        (r'\n\n_McClave et al_\s*\n\n\d+\s*\n\n', ' '),
        (r'_McClave et al_\s*\n', ''),
        (r'^\d{3}\s*$\n?', ''),
        (r'(?<=\w)\s*\n\n\d{3}\s*\n\n(?=\S)', ' '),
        (r'GRADE,.*?\n', ''),
        (r'Adapted from.*?\n', ''),

        # ── 8. remove stray ## headers
        (r'##\s+\*\*\[.*?\]\*\*\s*\n', ''),
        (r'##\s+\*\*[A-Z]\d+[a-z]?\..*?\n', ''),
        (r'##\s+\*\*H4\..*?\n', ''),
        (r'##\s+\*\*K2\..*?\n', ''),
        (r'##\s+\*\*L\d+[a-z]?\..*?\n', ''),
        (r'##\s+\*\*Q\d+\..*?\n', ''),
        (r'##\s+\*\*L3a\..*?\n', ''),

        # ── 9. remove empty headers
        (r'\n\n##\s*\n\n', '\n\n'),
        (r'##\s*\n', ''),
        (r'^\s*##\s*$', ''),

        # ── 10. remove citation numbers
        (r'\[\d+(?:[,;–\-]\d+)*\]', ''),

        # ── 11. strip markdown formatting
        (r'\*{1,2}', ''),
        (r'(?<!\w)_(?=\w)', ''),
        (r'(?<=\w)_(?!\w)', ''),

        # ── 12. whitespace cleanup
        (r'\n{3,}', '\n\n'),
        (r'^\s+$', ''),
    ]

    for pattern, replacement in patterns:
        if r'\|' in pattern:
            text = re.sub(pattern, replacement, text, flags=re.MULTILINE | re.DOTALL)
        else:
            text = re.sub(pattern, replacement, text, flags=re.MULTILINE)

    return text.strip()
```

</details>

<details><summary><code>knowledge_base[&#x27;A. Nutrition Assessment&#x27;]</code></summary>

```python
knowledge_base['A. Nutrition Assessment']
```

</details>

```text
'[ASPEN_2016_guidelines][A. Nutrition Assessment] _Question:_ Does the use of a nutrition risk indicator identify patients who will most likely benefit from nutrition therapy? \n\n**A1. Based on expert consensus, we suggest a determination of nutrition risk (eg, nutritional risk screening [NRS 2002], NUTRIC score) be performed on all patients admitted to the ICU for whom volitional** \n\n_Journal of Parenteral and Enteral Nutrition 40(2)_ \n\n162 \n\n## **Table 2.** Bundle Statements. \n\n- \x07Assess patients on admission to the intensive care unit (ICU) for nutrition risk, and calculate both energy and protein requirements to determine goals of nutrition therapy. \n\n- ● \x07Initiate enteral nutrition (EN) within 24−48 hours following the onset of critical illness and admission to the ICU, and increase to goals over the first week of ICU stay. \n\n- ● \x07Take steps as needed to reduce risk of aspiration or improve tolerance to gastric feeding (use prokinetic agent, continuous infusion, chlorhexidine mouthwash, elevate the head of bed, and divert level of feeding in the gastrointestinal tract). \n\n- ● Implement enteral feeding protocols with institution-specific strategies to promote delivery of EN. ● Do not use gastric residual volumes as part of routine care to monitor ICU patients receiving EN. ● Start parenteral nutrition early when EN is not feasible or sufficient in high-risk or poorly nourished patients. \n\n## **intake is anticipated to be insufficient. High nutrition risk identifies those patients most likely to benefit from early EN therapy.** \n\n_**Rationale:**_ Poor outcomes have been associated with inflammation generated by critical illness that leads to deterioration of nutrition status and malnutrition.[9] However, malnutrition in the critically ill has always been difficult to define. An international consensus group modified definitions to recognize the impact of inflammation. Objective measures of baseline nutrition status have been described by A.S.P.E.N. and the Academy of Nutrition and Dietetics.[10,11] However, nutrition risk is easily defined and more readily determined by evaluation of baseline nutrition status and assessment of disease severity. All hospitalized patients are required to undergo an initial nutrition screen within 48 hours of admission. However, patients at higher nutrition risk in an ICU setting require a full nutrition assessment. Many screening and assessment tools are used to evaluate nutrition status, such as the Mini Nutritional Assessment, the Malnutrition Universal Screening Tool, the Short Nutritional Assessment Questionnaire, the Malnutrition Screening Tool, and the Subjective Global Assessment.[12] However, only the NRS 2002 and the NUTRIC score determine both nutrition status and disease severity. Although both scoring systems were based on retrospective analysis, they have been used to define nutrition risk in RCTs in critically ill patients.[13–16] Patients at “risk” are defined by an NRS 2002 >3 and those at “high risk” with a score ≥ 5 or a NUTRIC score ≥ 5 (if interleukin-6 is not included, otherwise >6).[13–18] Interleukin-6 is rarely available as a component for the NUTRIC score; therefore, Heyland et al have shown that a NUTRIC score ≥ 5 still indicates high nutrition risk.[19] Two prospective nonrandomized studies show that patients at high nutrition risk are more likely to benefit from early EN with improved outcome (reduced nosocomial infection, total complications, and mortality) than patients at low nutrition risk.[13,18] While widespread use and supportive evidence are somewhat lacking to date, improvement in these scoring systems may increase their applicability in the future by providing guidance as to the role of EN and PN in the ICU. \n\n_Question:_ What additional tools, components, or surrogate markers provide useful information when performing nutrition assessments in critically ill adult patients? \n\n**A2. Based on expert consensus, we suggest that nutrition assessment include an evaluation of comorbid conditions, function of the gastrointestinal (GI) tract, and risk of aspiration. We suggest not using traditional nutrition indicators or surrogate markers, as they are not validated in critical care.** \n\n_**Rationale:**_ In the critical care setting, the traditional serum protein markers (albumin, prealbumin, transferrin, retinolbinding protein) are a reflection of the acute-phase response (increases in vascular permeability and reprioritization of hepatic protein synthesis) and do not accurately represent nutrition status in the ICU setting.[20] Anthropometrics are not reliable in assessment of nutrition status or adequacy of nutrition therapy.[21] Individual levels of calcitonin, C-reactive protein (CRP), interleukin-1, tumor necrosis factor (TNF), interleukin-6, and citrulline are still investigational and should not be used as surrogate markers. Ultrasound is emerging as a tool to expediently measure muscle mass and determine changes in muscle tissue at bedside in the ICU, given its ease of use and availability.[22,23] A computed tomography (CT) scan provides a precise quantification of skeletal muscle and adipose tissue depots; however, it is quite costly unless a scan taken for other purposes is used to determine body composition.[24,25] Both may be valuable future tools to incorporate into nutrition assessment; however, validation and reliability studies in ICU patients are still pending. Assessment of muscle function is still in its infancy. Its measurement, reproducibility, and applicability are still being validated for use in critically ill patients and may be of value in the future. \n\n_Question:_ What is the best method for determining energy needs in the critically ill adult patient? \n\n**A3a. We suggest that indirect calorimetry (IC) be used to determine energy requirements, when available and** \n\n_Journal of Parenteral and Enteral Nutrition 40(2)_ \n\n164 \n\n**in the absence of variables that affect the accuracy of measurement.** \n\n## **[Quality of Evidence: Very Low]** \n\n**A3b. Based on expert consensus, in the absence of IC, we suggest that a published predictive equation or a simplistic weight-based equation (25–30 kcal/kg/d) be used to determine energy requirements. (See section Q for obesity recommendations.)** \n\n_**Rationale:**_ Clinicians should determine energy requirements to establish the goals of nutrition therapy. Energy requirements may be calculated through simplistic formulas (25–30 kcal/ kg/d), published predictive equations, or IC. The applicability of IC may be limited at most institutions by availability and cost. Variables in the ICU that affect the timing and accuracy of IC measurements include the presence of air leaks or chest tubes, supplemental oxygen (eg, nasal cannula, bilevel positive airway pressure), ventilator settings (fractional inspiratory oxygen and positive end-expiratory pressure), continuous renal replacement therapy (CRRT), anesthesia, physical therapy, and excessive movement.[26] More than 200 predictive equations have been published in the literature, with accuracy rates ranging from 40%–75% when compared with IC, and no single equation emerges as being more accurate in an ICU.[27–32] Predictive equations are less accurate in obese and underweight patients.[33–36] Equations derived from testing hospital patients (Penn State, Ireton-Jones, Swinamer) are no more accurate than equations derived from testing normal volunteers (Harris-Benedict, Mifflin St Jeor).[37] The poor accuracy of predictive equations is related to many nonstatic variables affecting energy expenditure in the critically ill patient, such as weight, medications, treatments, and body temperature. The only advantage of using weight-based equations over other predictive equations is simplicity. However, in critically ill patients following aggressive volume resuscitation or in the presence of edema or anasarca, clinicians should use dry or usual body weight in these equations. \n\nAdditional energy provided by dextrose-containing fluids and lipid-based medications such as propofol should be accounted for when deriving nutrition therapy regimens to meet target energy goals. Achieving energy balance as guided by IC measurements compared with predictive equations may lead to more appropriate nutrition intake. \n\nWhile 2 RCTs[38,39] that met our inclusion criteria (with data from 161 patients) showed that higher mean intake of energy and protein was provided in IC-directed study patients compared with controls whose nutrition therapy was directed by predictive equations, issues with study design prevent a stronger recommendation for use of IC. In a study of burn patients, use of IC-directed nutrition therapy helped provide the minimal effective intake, avoiding the excesses of overfeeding seen in controls whose therapy was directed by the Curreri formula. Complications between groups (diarrhea and hyperglycemia) were no different, \n\nbut traditional outcome parameters were not evaluated.[38] A second study in general ICU patients used both EN and PN to meet target energy goals determined by IC measurement or a weightbased predictive equation (25 kcal/kg/d).[39] While the IC-directed energy goal was no different from the value obtained by predictive equation (1976 ± 468 vs 1838 ± 468 kcal/d, respectively; _P_ = .60), only study patients were monitored vigilantly by an ICU dietitian, while controls were managed by standard of care (less frequent ICU dietitian monitoring), which led to significantly more energy and protein per day in the study patients. The trend toward reduced mortality in study patients compared with controls (risk ratio [RR] = 0.63; 95% confidence interval [95% CI], 0.39–1.02; _P_ = .06) is difficult to reconcile in light of their increased morbidity with regard to ICU LOS (17.2 ± 14.6 vs 11.7 ± 8.4 days; _P_ = .04) and duration of mechanical ventilation (16.1 ± 14.7 vs 10.5 ± 8.3 days; _P_ = .03).[38,39] \n\nWhether measured by IC or estimated by predictive equations, energy expenditure should be reevaluated more than once per week, and strategies to optimize energy and protein intake should be used.[39,40] \n\n_Question:_ Should protein provision be monitored independently from energy provision in critically ill adult patients? \n\n## **A4. Based on expert consensus, we suggest an ongoing evaluation of adequacy of protein provision be performed.** \n\n_**Rationale:**_ In the critical care setting, protein appears to be the most important macronutrient for healing wounds, supporting immune function, and maintaining lean body mass. For most critically ill patients, protein requirements are proportionately higher than energy requirements and thus are not easily met by provision of routine enteral formulations (which have a high nonprotein calorie:nitrogen ratio [NPC:N]). Patients with suboptimal EN due to frequent interruptions may benefit from protein supplementation. The decision to add protein modules should be based on an ongoing assessment of adequacy of protein intake. Weight-based equations (eg, 1.2–2.0 g/kg/d) may be used to monitor adequacy of protein provision by comparing the amount of protein delivered with that prescribed, especially when nitrogen balance studies are not available to assess needs (see section C4).[41,42] Serum protein markers (albumin, prealbumin, transferrin, CRP) are not validated for determining adequacy of protein provision and should not be used in the critical care setting in this manner.[20,43]'
```

<details><summary><code>for key, content in knowledge_base.items():</code></summary>

```python
for key, content in knowledge_base.items():
    cleaned = clean_section(content)
    knowledge_base[key] = cleaned
```

</details>

<details><summary><code>knowledge_base[&#x27;A. Nutrition Assessment&#x27;]</code></summary>

```python
knowledge_base['A. Nutrition Assessment']
```

</details>

```text
'[ASPEN_2016_guidelines][A. Nutrition Assessment] Question: Does the use of a nutrition risk indicator identify patients who will most likely benefit from nutrition therapy? \n\nA1. Based on expert consensus, we suggest a determination of nutrition risk (eg, nutritional risk screening [NRS 2002], NUTRIC score) be performed on all patients admitted to the ICU for whom volitional \n\nintake is anticipated to be insufficient. High nutrition risk identifies those patients most likely to benefit from early EN therapy. \n\nRationale: Poor outcomes have been associated with inflammation generated by critical illness that leads to deterioration of nutrition status and malnutrition. However, malnutrition in the critically ill has always been difficult to define. An international consensus group modified definitions to recognize the impact of inflammation. Objective measures of baseline nutrition status have been described by A.S.P.E.N. and the Academy of Nutrition and Dietetics. However, nutrition risk is easily defined and more readily determined by evaluation of baseline nutrition status and assessment of disease severity. All hospitalized patients are required to undergo an initial nutrition screen within 48 hours of admission. However, patients at higher nutrition risk in an ICU setting require a full nutrition assessment. Many screening and assessment tools are used to evaluate nutrition status, such as the Mini Nutritional Assessment, the Malnutrition Universal Screening Tool, the Short Nutritional Assessment Questionnaire, the Malnutrition Screening Tool, and the Subjective Global Assessment. However, only the NRS 2002 and the NUTRIC score determine both nutrition status and disease severity. Although both scoring systems were based on retrospective analysis, they have been used to define nutrition risk in RCTs in critically ill patients. Patients at “risk” are defined by an NRS 2002 >3 and those at “high risk” with a score ≥ 5 or a NUTRIC score ≥ 5 (if interleukin-6 is not included, otherwise >6). Interleukin-6 is rarely available as a component for the NUTRIC score; therefore, Heyland et al have shown that a NUTRIC score ≥ 5 still indicates high nutrition risk. Two prospective nonrandomized studies show that patients at high nutrition risk are more likely to benefit from early EN with improved outcome (reduced nosocomial infection, total complications, and mortality) than patients at low nutrition risk. While widespread use and supportive evidence are somewhat lacking to date, improvement in these scoring systems may increase their applicability in the future by providing guidance as to the role of EN and PN in the ICU. \n\nQuestion: What additional tools, components, or surrogate markers provide useful information when performing nutrition assessments in critically ill adult patients? \n\nA2. Based on expert consensus, we suggest that nutrition assessment include an evaluation of comorbid conditions, function of the gastrointestinal (GI) tract, and risk of aspiration. We suggest not using traditional nutrition indicators or surrogate markers, as they are not validated in critical care. \n\nRationale: In the critical care setting, the traditional serum protein markers (albumin, prealbumin, transferrin, retinolbinding protein) are a reflection of the acute-phase response (increases in vascular permeability and reprioritization of hepatic protein synthesis) and do not accurately represent nutrition status in the ICU setting. Anthropometrics are not reliable in assessment of nutrition status or adequacy of nutrition therapy. Individual levels of calcitonin, C-reactive protein (CRP), interleukin-1, tumor necrosis factor (TNF), interleukin-6, and citrulline are still investigational and should not be used as surrogate markers. Ultrasound is emerging as a tool to expediently measure muscle mass and determine changes in muscle tissue at bedside in the ICU, given its ease of use and availability. A computed tomography (CT) scan provides a precise quantification of skeletal muscle and adipose tissue depots; however, it is quite costly unless a scan taken for other purposes is used to determine body composition. Both may be valuable future tools to incorporate into nutrition assessment; however, validation and reliability studies in ICU patients are still pending. Assessment of muscle function is still in its infancy. Its measurement, reproducibility, and applicability are still being validated for use in critically ill patients and may be of value in the future. \n\nQuestion: What is the best method for determining energy needs in the critically ill adult patient? \n\nA3a. We suggest that indirect calorimetry (IC) be used to determine energy requirements, when available and in the absence of variables that affect the accuracy of measurement. \n\nA3b. Based on expert consensus, in the absence of IC, we suggest that a published predictive equation or a simplistic weight-based equation (25–30 kcal/kg/d) be used to determine energy requirements. (See section Q for obesity recommendations.) \n\nRationale: Clinicians should determine energy requirements to establish the goals of nutrition therapy. Energy requirements may be calculated through simplistic formulas (25–30 kcal/ kg/d), published predictive equations, or IC. The applicability of IC may be limited at most institutions by availability and cost. Variables in the ICU that affect the timing and accuracy of IC measurements include the presence of air leaks or chest tubes, supplemental oxygen (eg, nasal cannula, bilevel positive airway pressure), ventilator settings (fractional inspiratory oxygen and positive end-expiratory pressure), continuous renal replacement therapy (CRRT), anesthesia, physical therapy, and excessive movement. More than 200 predictive equations have been published in the literature, with accuracy rates ranging from 40%–75% when compared with IC, and no single equation emerges as being more accurate in an ICU. Predictive equations are less accurate in obese and underweight patients. Equations derived from testing hospital patients (Penn State, Ireton-Jones, Swinamer) are no more accurate than equations derived from testing normal volunteers (Harris-Benedict, Mifflin St Jeor). The poor accuracy of predictive equations is related to many nonstatic variables affecting energy expenditure in the critically ill patient, such as weight, medications, treatments, and body temperature. The only advantage of using weight-based equations over other predictive equations is simplicity. However, in critically ill patients following aggressive volume resuscitation or in the presence of edema or anasarca, clinicians should use dry or usual body weight in these equations. \n\nAdditional energy provided by dextrose-containing fluids and lipid-based medications such as propofol should be accounted for when deriving nutrition therapy regimens to meet target energy goals. Achieving energy balance as guided by IC measurements compared with predictive equations may lead to more appropriate nutrition intake. \n\nWhile 2 RCTs that met our inclusion criteria (with data from 161 patients) showed that higher mean intake of energy and protein was provided in IC-directed study patients compared with controls whose nutrition therapy was directed by predictive equations, issues with study design prevent a stronger recommendation for use of IC. In a study of burn patients, use of IC-directed nutrition therapy helped provide the minimal effective intake, avoiding the excesses of overfeeding seen in controls whose therapy was directed by the Curreri formula. Complications between groups (diarrhea and hyperglycemia) were no different, \n\nbut traditional outcome parameters were not evaluated. A second study in general ICU patients used both EN and PN to meet target energy goals determined by IC measurement or a weightbased predictive equation (25 kcal/kg/d). While the IC-directed energy goal was no different from the value obtained by predictive equation (1976 ± 468 vs 1838 ± 468 kcal/d, respectively; P = .60), only study patients were monitored vigilantly by an ICU dietitian, while controls were managed by standard of care (less frequent ICU dietitian monitoring), which led to significantly more energy and protein per day in the study patients. The trend toward reduced mortality in study patients compared with controls (risk ratio [RR] = 0.63; 95% confidence interval [95% CI], 0.39–1.02; P = .06) is difficult to reconcile in light of their increased morbidity with regard to ICU LOS (17.2 ± 14.6 vs 11.7 ± 8.4 days; P = .04) and duration of mechanical ventilation (16.1 ± 14.7 vs 10.5 ± 8.3 days; P = .03). \n\nWhether measured by IC or estimated by predictive equations, energy expenditure should be reevaluated more than once per week, and strategies to optimize energy and protein intake should be used. \n\nQuestion: Should protein provision be monitored independently from energy provision in critically ill adult patients? \n\nRationale: In the critical care setting, protein appears to be the most important macronutrient for healing wounds, supporting immune function, and maintaining lean body mass. For most critically ill patients, protein requirements are proportionately higher than energy requirements and thus are not easily met by provision of routine enteral formulations (which have a high nonprotein calorie:nitrogen ratio [NPC:N]). Patients with suboptimal EN due to frequent interruptions may benefit from protein supplementation. The decision to add protein modules should be based on an ongoing assessment of adequacy of protein intake. Weight-based equations (eg, 1.2–2.0 g/kg/d) may be used to monitor adequacy of protein provision by comparing the amount of protein delivered with that prescribed, especially when nitrogen balance studies are not available to assess needs (see section C4). Serum protein markers (albumin, prealbumin, transferrin, CRP) are not validated for determining adequacy of protein provision and should not be used in the critical care setting in this manner.'
```

<details><summary><code>knowledge_base[&#x27;B. Initiate EN&#x27;]</code></summary>

```python
knowledge_base['B. Initiate EN']
```

</details>

```text
'[ASPEN_2016_guidelines][B. Initiate EN] Question: What is the benefit of early EN in critically ill adult patients compared with withholding or delaying this therapy? \n\nB1. We recommend that nutrition support therapy in the form of early EN be initiated within 24–48 hours in the critically ill patient who is unable to maintain volitional intake. \n\nRationale: EN supports the functional integrity of the gut by maintaining tight junctions between the intraepithelial cells, stimulating blood flow, and inducing the release of trophic endogenous agents (eg, cholecystokinin, gastrin, bombesin, and bile salts). EN maintains structural integrity by maintaining villous height and supporting the mass of secretory IgAproducing immunocytes (B cells and plasma cells) that compose the gut-associated lymphoid tissue (GALT) and in turn contribute to mucosal-associated lymphoid tissue at distant sites such as the lungs, liver, and kidneys. \n\nAdverse change in gut permeability from loss of functional integrity is a dynamic phenomenon that is time dependent (channels opening within hours of the major insult or injury). The consequences of the permeability changes include increased bacterial challenge (engagement of GALT with enteric organisms), risk for systemic infection, and greater likelihood of multiple-organ dysfunction syndrome. As disease severity worsens, increases in gut permeability are amplified, and the enteral route of feeding is more likely to favorably impact outcome parameters of infection, organ failure, and hospital LOS. \n\nThe specific reasons for providing EN are to maintain gut integrity, modulate stress and the systemic immune response, and attenuate disease severity. Additional \n\nend points of EN therapy may include use of the gut as a conduit for the delivery of immune-modulating agents and use of enteral formulations as an effective means for stress ulcer prophylaxis. \n\nThree previous meta-analyses aggregated data from RCTs comparing early versus delayed EN. One meta-analysis of 8 trials by Heyland et al showed a trend toward reduced mortality (RR = 0.52; 95% CI, 0.25–1.08; P = .08) when EN was started within 48 hours, compared with delayed initiation of EN started after that point. A second meta-analysis of 12 trials by Marik et al showed significant reductions in infectious morbidity (RR = 0.45; 95% CI, 0.30–0.66; P = .00006) and hospital LOS (mean, 2.2 days; 95% CI, 0.81–3.63 days; P = .001) when early EN was started on average within 36 hours of ICU admission. A third meta-analysis of 6 trials by Doig et al showed a significant reduction in pneumonia (odds ratio [OR] = 0.31; 95% CI, 0.12–0.78; P = .01) and mortality (OR = 0.34; 95% CI, 0.14–0.85; P = .02) but no difference in multiple-organ failure (MOF) when early EN was started within 24 hours of admission to the ICU, compared with EN started after that point. \n\nOf an updated meta-analysis of 21 RCTs that met our inclusion criteria comparing the provision of early EN versus delayed EN, all reported on mortality (Figure 1), with 13 reporting on infection (Figure 2). Provision of early EN was associated with a significant reduction in mortality (RR = 0.70; 95% CI, 0.49–1.00; P = .05) and infectious morbidity (RR = 0.74; 95% CI, 0.58–0.93; P = .01), compared with withholding early EN (delayed EN or STD). \n\nJournal of Parenteral and Enteral Nutrition 40(2)_  Question: Is there a difference in outcome between the use of EN or PN for adult critically ill patients? \n\nRationale: In the majority of critically ill patients, it is practical and safe to use EN instead of PN. The beneficial effects of EN compared with PN are well documented in numerous RCTs involving a variety of patient populations in critical illness, including trauma, burns, head injury, major surgery, and acute pancreatitis. While few studies have shown a differential effect on mortality, the most consistent outcome effect from EN is a reduction in infectious morbidity (generally, pneumonia and central line infections in most patient populations; specifically, abdominal abscess in trauma patients) and ICU LOS. \n\nSix previous meta-analyses comparing EN with PN showed significant reductions in infectious morbidity with use of EN. Noninfective complications (risk difference = 4.9; 95% CI, 0.3–9.5; P = .04) and reduced hospital LOS (weighted mean difference [WMD] = 1.20 days; 95% CI, 0.38–2.03; P = .004) were seen with use of EN compared with PN in one of the meta-analyses by Peter et al. Five of the meta-analyses showed no difference in mortality between the 2 routes of nutrition support therapy. One meta-analysis by Simpson and Doig showed a significantly lower mortality (RR = 0.51; 95% CI, 0.27–0.97; P = .04) despite a significantly higher incidence of infectious complications (RR = 1.66; 95% CI, 1.09– 2.51; P = .02) with use of PN compared with EN. \n\nIn 12 studies representing 618 patients that met our inclusion criteria, 9 reported on infection (Figure 3), which was shown to be significantly less with EN than PN (RR = 0.56; 95% CI, 0.39–0.79; P < .00001). ICU LOS also was shorter with EN compared with PN by nearly 1 full day (WMD = −0.82 days; 95% CI, −1.29 to −0.34; P = .0007). Hospital LOS and mortality were not significantly different. These differences in outcome from the separate routes of feeding largely reflect findings from older studies and may diminish in the future with improvements in glycemic control, protocolized medical management, and new lipid emulsions. \n\nQuestion: Is the clinical evidence of contractility (bowel sounds, flatus) required prior to initiating EN in critically ill adult patients? \n\nB3. Based on expert consensus, we suggest that, in the majority of MICU and SICU patient populations, while GI contractility factors should be evaluated when initiating EN, overt signs of contractility should not be required prior to initiation of EN. \n\nRationale: The literature supports the concept that bowel sounds and evidence of bowel function (ie, passing flatus or stool) are not required for initiation of EN. GI dysfunction in the ICU setting occurs in 30%–70% of patients, depending on the diagnosis, premorbid condition, ventilation mode, medications, and metabolic state. \n\nProposed mechanisms of ICU and postoperative GI dysfunction are related to mucosal barrier disruption, altered motility, atrophy of the mucosa, and reduced mass of GALT. GI intolerance has been variably defined (eg, absence or abnormal bowel sounds, vomiting, bowel dilatation, diarrhea, GI bleeding, high gastric residual volumes [GRVs]) and appears to occur in up to 50% of patients on mechanical ventilation. Bowel sounds are indicative only of contractility and do not necessarily relate to mucosal integrity, barrier function, or absorptive capacity. \n\nThe argument for initiating EN regardless of the extent of audible bowel sounds is based on studies (most of which involve critically ill surgical patients) reporting the feasibility and safety of EN within the initial 36–48 hours of admission to the ICU. \n\nNonetheless, reduced or absent bowel sounds may reflect greater disease severity and worsened prognosis. Patients with normal bowel sounds have been shown to have lower ICU mortality than those with hypoactive or absent bowel sounds \n\n(11.3% vs 22.6% vs 36.0%, respectively). ICU LOS has been shown to increase with greater number of symptoms of GI intolerance (2.9 days when asymptomatic vs up to 16.8 days with 4 symptoms of intolerance). Not surprising, success of EN delivery is reduced with a greater number of symptoms of GI intolerance. A greater number of signs of intolerance may warrant increased vigilance as EN is started and may necessitate further clinical evaluation. \n\nQuestion: What is the preferred level of infusion of EN within the GI tract for critically ill patients? How does the level of infusion of EN affect patient outcomes? \n\nB4a. We recommend that the level of infusion be diverted lower in the GI tract in those critically ill patients at high risk for aspiration (see section D4) or those who have shown intolerance to gastric EN. \n\nB4b. Based on expert consensus we suggest that, in most critically ill patients, it is acceptable to initiate EN in the stomach. \n\nRationale: Initiating EN therapy in the stomach is technically easier and may decrease the time to initiation of EN. The choice of level of infusion within the GI tract (ie, whether the tip of the feeding tube is in the stomach, different segments of the duodenum [D1, D2, D3, or D4], or the jejunum) may be determined by patient selection within ICU practitioners’ institutional framework (ease and feasibility of placing small bowel enteral access devices, institutional policies, and protocols). \n\nIn the largest multicenter RCT to compare gastric versus small bowel EN in critically ill patients, Davies et al found no difference in clinical outcomes between groups, including LOS, mortality, nutrient delivery, and incidence of pneumonia. Aggregating the data from the RCTs that met our inclusion criteria, 6 trials reported on improved nutrient delivery with small bowel feedings (WMD = 11.06%; 95% CI, 5.82–16.30%; P < .00001) (Figure 4), and 12 trials demonstrated a reduced risk of pneumonia compared with gastric EN (RR = 0.75; 95% \n\nJournal of Parenteral and Enteral Nutrition 40(2)_  CI, 0.60–0.93; P = .01) (Figure 5). Although small bowel EN decreases the risk of pneumonia, there is no difference in mortality or LOS between small bowel and gastric EN. Therefore, if timely obtainment of small bowel enteral access device is not feasible, early EN via the gastric route may provide more benefit than delaying feeding initiation while awaiting small bowel access. \n\nQuestion: Is EN safe during periods of hemodynamic instability in adult critically ill patients? \n\nB5. Based on expert consensus, we suggest that in the setting of hemodynamic compromise or instability, EN should be withheld until the patient is fully resuscitated and/or stable. Initiation/reinitiation of EN may be considered with caution in patients undergoing withdrawal of vasopressor support. \n\nWhile EN may be provided with caution to patients on chronic, stable low doses of vasopressors, EN should be withheld in patients who are hypotensive (mean arterial blood pressure <50 mm Hg), in patients for whom catecholamine agents (eg, norepinephrine, phenylephrine, epinephrine, dopamine) are being initiated, or in patients for whom escalating doses are required to maintain hemodynamic stability. \n\nFor patients on vasopressor therapy receiving EN, any signs of intolerance (abdominal distention, increasing nasogastric [NG] tube output or GRVs, decreased passage of stool and flatus, hypoactive bowel sounds, increasing metabolic acidosis and/or base deficit) should be closely scrutinized as possible early signs of gut ischemia, and EN should be held until symptoms and interventions stabilize.'
```

<details><summary><code># Validation</code></summary>

```python
# Validation
print("KNOWLEDGE BASE VALIDATION")
print("=" * 50)
print(f"Total documents: {len(knowledge_base)}")

total_words = 0
for doc_name, content in knowledge_base.items():
    word_count = len(content.split())
    total_words += word_count
    status = "Pass" if word_count >= 150 else "Too short!"
    print(f"  {doc_name}: {word_count} words {status}")

print(f"\nTotal words: {total_words}")
print(f"Average words per document: {total_words // len(knowledge_base)}")

if len(knowledge_base) >= 8:
    print("\nDocument count requirement met!")
else:
    print(f"\nYou need at least 8 documents. Currently have: {len(knowledge_base)}")
```

</details>

```text
KNOWLEDGE BASE VALIDATION
==================================================
Total documents: 22
  A. Nutrition Assessment: 1477 words Pass
  B. Initiate EN: 1707 words Pass
  C. Dosing of EN: 1256 words Pass
  D. Monitoring Tolerance and Adequacy of EN: 2558 words Pass
  E. Selection of Appropriate Enteral Formulation: 1222 words Pass
  F. Adjunctive Therapy: 1334 words Pass
  G. When to Use PN: 1174 words Pass
  H. When Indicated, Maximize Efficacy of PN: 2694 words Pass
  I. Pulmonary Failure: 547 words Pass
  J. Renal Failure: 372 words Pass
  K. Hepatic Failure: 714 words Pass
  L. Acute Pancreatitis: 1741 words Pass
  M. Surgical Subsets: 3 words Too short!
  Trauma: 613 words Pass
  Traumatic Brain Injury: 161 words Pass
  Open Abdomen: 825 words Pass
  Burns: 702 words Pass
  N. Sepsis: 1568 words Pass
  O. Postoperative Major Surgery (SICU Admission Expected): 2272 words Pass
  P. Chronically Critically Ill: 332 words Pass
  Q. Obesity in Critical Illness: 2731 words Pass
  R. Nutrition Therapy End-of-Life Situations: 118 words Too short!

Total words: 26121
Average words per document: 1187

Document count requirement met!
```

---

## Part 4: Document Chunking

Implement a chunking function and processing the knowledge base.

<details><summary><code>def chunk_document(</code></summary>

```python
def chunk_document(
    text: str,
    chunk_size: int = 650,
    chunk_overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks, respecting sentence boundaries.

    Args:
        text: The text to chunk
        chunk_size: Target size for each chunk (in characters)
        chunk_overlap: Number of characters to overlap between chunks

    Returns:
        List of text chunks
    """
    # Clean the text
    text = text.strip()
    text = re.sub(r'\n+', ' ', text)  # Replace newlines with spaces
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace

    # Split into sentences (simple approach)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If adding this sentence would exceed chunk size
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Start new chunk with overlap from previous chunk
            overlap_start = max(0, len(current_chunk) - chunk_overlap)
            current_chunk = current_chunk[overlap_start:] + " " + sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence

    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

print("✅ Chunking function defined")


# Test your chunking function
test_text = list(knowledge_base.values())[1]
test_chunks = chunk_document(test_text)

print("🔍 CHUNKING TEST")
print("=" * 50)
print(f"Original document length: {len(test_text)} characters")
print(f"Number of chunks created: {len(test_chunks)}")
print(f"\nChunk previews:")
print(test_chunks[0][:] + "...." if len(test_chunks) > 0 else "No chunks created")
print("")
print(test_chunks[1][:] + "...." if len(test_chunks) > 0 else "No chunks created")
```

</details>

```text
✅ Chunking function defined
🔍 CHUNKING TEST
==================================================
Original document length: 10920 characters
Number of chunks created: 24

Chunk previews:
[ASPEN_2016_guidelines][B. Initiate EN] Question: What is the benefit of early EN in critically ill adult patients compared with withholding or delaying this therapy? B1. We recommend that nutrition support therapy in the form of early EN be initiated within 24–48 hours in the critically ill patient who is unable to maintain volitional intake. Rationale: EN supports the functional integrity of the gut by maintaining tight junctions between the intraepithelial cells, stimulating blood flow, and inducing the release of trophic endogenous agents (eg, cholecystokinin, gastrin, bombesin, and bile salts).....

g the release of trophic endogenous agents (eg, cholecystokinin, gastrin, bombesin, and bile salts). EN maintains structural integrity by maintaining villous height and supporting the mass of secretory IgAproducing immunocytes (B cells and plasma cells) that compose the gut-associated lymphoid tissue (GALT) and in turn contribute to mucosal-associated lymphoid tissue at distant sites such as the lungs, liver, and kidneys. Adverse change in gut permeability from loss of functional integrity is a dynamic phenomenon that is time dependent (channels opening within hours of the major insult or injury).....
```

<details><summary><code># Configure chunking parameters</code></summary>

```python
# Configure chunking parameters
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128

# Process all documents
all_chunks = []
all_metadatas = []
all_ids = []

for doc_name, document in knowledge_base.items():
  match = re.match(r'(\[.*?\]\[.*?\])', document.strip())
  doc_title = match.group(1) if match else doc_name

  match = re.match(r'(\[.*?\])', document.strip())
  source = match.group(1) if match else doc_name

  chunks = chunk_document(document, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

  for i, chunk in enumerate(chunks):
    contextualized_chunk = chunk if i == 0 else f"{doc_title} {chunk}"

    all_chunks.append(contextualized_chunk)
    all_metadatas.append({
        "source":source,
        "title":doc_title,
        "chunk_index": i
    })
    all_ids.append(f"{doc_title}_chunk_{i}")

# Validation
print("CHUNKING RESULTS")
print("=" * 50)
print(f"Total chunks created: {len(all_chunks)}")
print(f"Chunk size setting: {CHUNK_SIZE}")
print(f"Chunk overlap setting: {CHUNK_OVERLAP}")
```

</details>

```text
CHUNKING RESULTS
==================================================
Total chunks created: 221
Chunk size setting: 1024
Chunk overlap setting: 128
```

<details><summary><code>all_chunks[3], all_metadatas[3], all_ids[3]</code></summary>

```python
all_chunks[3], all_metadatas[3], all_ids[3]
```

</details>

```text
('[ASPEN_2016_guidelines][A. Nutrition Assessment] ese scoring systems may increase their applicability in the future by providing guidance as to the role of EN and PN in the ICU. Question: What additional tools, components, or surrogate markers provide useful information when performing nutrition assessments in critically ill adult patients? A2. Based on expert consensus, we suggest that nutrition assessment include an evaluation of comorbid conditions, function of the gastrointestinal (GI) tract, and risk of aspiration. We suggest not using traditional nutrition indicators or surrogate markers, as they are not validated in critical care. Rationale: In the critical care setting, the traditional serum protein markers (albumin, prealbumin, transferrin, retinolbinding protein) are a reflection of the acute-phase response (increases in vascular permeability and reprioritization of hepatic protein synthesis) and do not accurately represent nutrition status in the ICU setting.',
 {'source': '[ASPEN_2016_guidelines]',
  'title': '[ASPEN_2016_guidelines][A. Nutrition Assessment]',
  'chunk_index': 3},
 '[ASPEN_2016_guidelines][A. Nutrition Assessment]_chunk_3')
```

### Chunking Strategy Justification

**Rationale of Chunk Size**

In the ASPEN 2016 guidelines, clincal questions and scenarios are generally 10-20 words (100-200 characters) long. Clincal guideance paragraphs are generally 200-300 words (1500-3000 characters long).

A chunk size of 1200 allows for the entire clinical question to be included in the inital chuck of each document along with crutial clinical context. As we are using medical guidelines, sentences are generally longer and interconnected. Using a chunk size of 1200 and overlap of 170 allows us a more coherent context window will keeping embeddings focused.

**Rationale of Overlap Amount**

We used a larger overlap of 170 to account for the possibility of longer clinical rationales (i.e. considerations of immune modulating formulas in diifferent populatitons). A larger overlap ensures that clinical reasoning between connected chunks is preserved.   

**Additional Modifications**

Originally a 53 page document, we split the paper into seperate documents based on condition type. Citations and tables were also removed. We also elected to use `"abhinand/MedEmbed-small-v0.1"` to better embed medical terminology.

---

## Part 5: Vector Database Setup

Creating a ChromaDB collection and add all chunks with embeddings.

<details><summary><code># Initialize ChromaDB client</code></summary>

```python
# Initialize ChromaDB client
chroma_client = chromadb.Client()

# Create embedding function
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="abhinand/MedEmbed-small-v0.1"
)

# embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
#     model_name="abhinand/MedEmbed-base-v0.1"  # upgrade from small
# )

# Include metadata describing your knowledge base

collection = chroma_client.create_collection(
    name="ICU-nutritition-guidelines-db",
    embedding_function=embedding_function,
    metadata={"description": "database of best practice for nutrititon intervention in the adult ICU patient"} 
)

print("Collection created!")
```

</details>

```text
model.safetensors: downloading bytes:           |  0.00B
Collection created!
```

<details><summary><code># Add chunks to the collection</code></summary>

```python
# Add chunks to the collection
collection.add(
    documents=all_chunks,
    metadatas=all_metadatas,
    ids=all_ids
)

print(f"✅ Added {collection.count()} chunks to the database")
```

</details>

```text
✅ Added 221 chunks to the database
```

<details><summary><code># Test retrieval with a sample query</code></summary>

```python
# Test retrieval with a sample query
test_query = "Calorie needs for the Acute Respiratory Distress Syndrome (ARDS) patient?"

results = collection.query(
    query_texts=[test_query],
    n_results=6
)

print(f"Test Query: \"{test_query}\"")
print("\nTop 3 Retrieved Chunks:")
for i in range(len(results['documents'][0])):
    print(f"\n{i+1}. Source: {results['metadatas'][0][i].get('source', 'Unknown')}")
    print(f"   Preview: {results['documents'][0][i][:]}...")
```

</details>

```text
Test Query: "Calorie needs for the Acute Respiratory Distress Syndrome (ARDS) patient?"

Top 3 Retrieved Chunks:

1. Source: [ASPEN_2016_guidelines]
   Preview: [ASPEN_2016_guidelines][Trauma] onged stays in the ICU and should undergo timely nutrition reassessment. Energy requirements vary depending on numerous factors. Resting energy expenditure (REE) peaks over 4–5 days but continues to remain high for 9–12 days (with some elevation in energy expenditure persisting for over 21 days). Approximately 16% of total body protein is lost in the first 21 days, with 67% of that protein loss coming from skeletal muscle alone. Energy goals should be in the range of 20–35 kcal/kg/d, depending on the phase of trauma. Lower energy provision is suggested early in the resuscitative phase, with liberalization of energy delivery as the patient enters into the rehabilitation phase. Requirements for protein are similar for other ICU patients but may be at the higher end of the provision range, from 1.2–2 g/kg/d (see section C4). Question: Should immune-modulation formulas be used routinely to improve outcomes in a patient with severe trauma? M1b....

2. Source: [ASPEN_2016_guidelines]
   Preview: [ASPEN_2016_guidelines][A. Nutrition Assessment] e tools to incorporate into nutrition assessment; however, validation and reliability studies in ICU patients are still pending. Assessment of muscle function is still in its infancy. Its measurement, reproducibility, and applicability are still being validated for use in critically ill patients and may be of value in the future. Question: What is the best method for determining energy needs in the critically ill adult patient? A3a. We suggest that indirect calorimetry (IC) be used to determine energy requirements, when available and in the absence of variables that affect the accuracy of measurement. A3b. Based on expert consensus, in the absence of IC, we suggest that a published predictive equation or a simplistic weight-based equation (25–30 kcal/kg/d) be used to determine energy requirements. (See section Q for obesity recommendations.) Rationale: Clinicians should determine energy requirements to establish the goals of nutrition therapy....

3. Source: [ASPEN_2016_guidelines]
   Preview: [ASPEN_2016_guidelines][J. Renal Failure] ICU patient, the underlying disease process, preexisting comorbidities, and current complications should be taken into account. In the absence of IC, no one predictive equation is better than another in AKI. Experts agree on using usual body weight for normal weight patients and ideal body weight for obese and critically ill patients. Energy needs can be determined by IC, published predictive equations, or a simplistic weight-based equation (25–30 kcal/kg/d). Specialty formulations lower in certain electrolytes (eg, phosphate and potassium) than standard products may be beneficial in ICU patients with AKI. Question: In adult critically ill patients with AKI receiving hemodialysis or CRRT, what are appropriate targets for protein intake to support increased nitrogen losses? J2. We recommend that patients receiving frequent hemodialysis or CRRT receive increased protein, up to a maximum of 2.5 g/kg/d....

4. Source: [ASPEN_2016_guidelines]
   Preview: [ASPEN_2016_guidelines][N. Sepsis] /d) for the initial phase of sepsis, advancing as tolerated after 24–48 hours to >80% of target energy goal over the first week. We suggest delivery of 1.2–2 g protein/kg/d. ‡References 219, 220, 225, 226, 230, 231, 288, 409, 410 Rationale: Wide variability in energy expenditure has been documented in advanced septic shock. For this reason, IC is recommended, if available, for baseline energy expenditure measurement, with follow-up measurement every 4 days. If IC is not available or patient conditions do not allow for it (eg, Fio2 >0.60), then simplistic weight-based equations (25 kcal/kg/d) or published equations may be used for predicting energy expenditure. In a cohort of patients with SIRS, sepsis, and septic shock, estimates from the Harris-Benedict and Schofield published equations correlated well with energy expenditure measured by IC (all results within 8% of each other). Observational studies suggest that provision of a range of 25%–66% of calculated energy requirements may be optimal....

5. Source: [ASPEN_2016_guidelines]
   Preview: [ASPEN_2016_guidelines][Q. Obesity in Critical Illness] or all classes of obesity, the goal of the EN regimen should not exceed 65%–70% of target energy requirements as measured by IC. If IC is unavailable, we suggest using the weight-based equation 11–14 kcal/kg actual body weight per day for patients with BMI in the range of 30–50 and 22–25 kcal/kg ideal body weight per day for patients with BMI >50. We suggest that protein should be provided in a range from 2.0 g/kg ideal body weight per day for patients with BMI of 30–40 up to 2.5 g/kg ideal body weight per day for patients with BMI ≥ 40. Rationale: Achieving some degree of weight loss may increase insulin sensitivity, facilitate nursing care, and reduce risk of comorbidities. Providing 60%–70% of caloric requirements promotes steady weight loss. A retrospective study by Choban et al indicated that provision of protein at a dose of 2.0 g/kg ideal body weight per day was insufficient for achieving neutral nitrogen balance when BMI is >40....

6. Source: [ASPEN_2016_guidelines]
   Preview: [ASPEN_2016_guidelines][I. Pulmonary Failure] Question: What is the optimal carbohydrate/fat ratio for the adult ICU patient with pulmonary failure? I1. We suggest that specialty high-fat/low-carbohydrate formulations designed to manipulate the respiratory quotient and reduce CO2 production not be used in ICU patients with acute respiratory failure (not to be confused with recommendation E3). ## [Quality of Evidence: Very Low] Rationale: An early, very small trial (20 patients) showed that use of a high-fat/low-carbohydrate enteral formulation in patients with respiratory failure reduced duration of mechanical ventilation, compared with a standard formulation. However, these findings could not be reproduced in a subsequent larger RCT (50 patients) of similar type. Results from uncontrolled studies would suggest that increasing the composite macronutrient ratio of fat to carbohydrate becomes clinically significant in lowering CO2 production only in the ICU patient who is being overfed....
```

<details><summary><code>results = collection.query(</code></summary>

```python
results = collection.query(
    query_texts=["Calorie needs for the Acute Respiratory Distress Syndrome (ARDS) patient?"],
    n_results=6  # get more results
)

# print all results with their distances
for i, (doc, distance, metadata) in enumerate(zip(
    results['documents'][0],
    results['distances'][0],
    results['metadatas'][0]
)):
    print(f"{i+1}. Distance: {distance:.4f} | {metadata['title']}")
    print(f"   {doc[:100]}")
    print()
```

</details>

```text
1. Distance: 0.2166 | [ASPEN_2016_guidelines][Trauma]
   [ASPEN_2016_guidelines][Trauma] onged stays in the ICU and should undergo timely nutrition reassessm

2. Distance: 0.2247 | [ASPEN_2016_guidelines][A. Nutrition Assessment]
   [ASPEN_2016_guidelines][A. Nutrition Assessment] e tools to incorporate into nutrition assessment; h

3. Distance: 0.2270 | [ASPEN_2016_guidelines][J. Renal Failure]
   [ASPEN_2016_guidelines][J. Renal Failure] ICU patient, the underlying disease process, preexisting c

4. Distance: 0.2282 | [ASPEN_2016_guidelines][N. Sepsis]
   [ASPEN_2016_guidelines][N. Sepsis] /d) for the initial phase of sepsis, advancing as tolerated after

5. Distance: 0.2289 | [ASPEN_2016_guidelines][Q. Obesity in Critical Illness]
   [ASPEN_2016_guidelines][Q. Obesity in Critical Illness] or all classes of obesity, the goal of the E

6. Distance: 0.2295 | [ASPEN_2016_guidelines][I. Pulmonary Failure]
   [ASPEN_2016_guidelines][I. Pulmonary Failure] Question: What is the optimal carbohydrate/fat ratio f
```

---

## Part 6: Building the RAG Pipeline

<details><summary><code># Load the extractive question-answering model</code></summary>

```python
# Load the extractive question-answering model
# NOTE: this model is a SQuAD extractive QA model -- it must NOT be loaded as "text-generation"
MODEL_NAME = "dmis-lab/biobert-large-cased-v1.1-squad"
device = 0 if torch.cuda.is_available() else -1

try:
    qa_model = pipeline("question-answering", model=MODEL_NAME, tokenizer=MODEL_NAME, device=device)
except (KeyError, ValueError) as e:
    # Fallback for transformers versions without the "question-answering" pipeline task
    print(f"[INFO] QA pipeline unavailable ({e}); using manual extractive QA")
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    _tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    _mdl = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME).to("cuda" if device == 0 else "cpu").eval()

    def qa_model(question: str, context: str, max_length: int = 384, stride: int = 128, max_answer_len: int = 60):
        enc = _tok(question, context, truncation="only_second", max_length=max_length, stride=stride,
                   return_overflowing_tokens=True, return_offsets_mapping=True, padding="max_length",
                   return_tensors="pt")
        offsets = enc.pop("offset_mapping"); enc.pop("overflow_to_sample_mapping", None)
        enc = {k: v.to(_mdl.device) for k, v in enc.items()}
        with torch.no_grad():
            out = _mdl(**enc)
        best = {"answer": "", "score": 0.0}
        for w in range(out.start_logits.shape[0]):
            seq_ids = _tok(question, context, truncation="only_second", max_length=max_length, stride=stride,
                           return_overflowing_tokens=True).sequence_ids(w)
            ctx_mask = torch.tensor([s == 1 for s in seq_ids] + [False] * (max_length - len(seq_ids)), device=_mdl.device)
            s = out.start_logits[w].masked_fill(~ctx_mask, -1e4).softmax(-1)
            e = out.end_logits[w].masked_fill(~ctx_mask, -1e4).softmax(-1)
            scores = torch.triu(s[:, None] * e[None, :]) - torch.triu(torch.ones_like(s[:, None] * e[None, :]), diagonal=max_answer_len)
            idx = scores.argmax(); i, j = divmod(idx.item(), scores.shape[1]); sc = scores[i, j].item()
            if sc > best["score"]:
                best = {"answer": context[offsets[w][i][0]:offsets[w][j][1]], "score": sc}
        return best

print("QA model loaded!")
```

</details>

```text
[INFO] QA pipeline unavailable ('The task question-answering has been removed from transformers. Use the `text-generation` pipeline with a chat model instead. See the migration guide for more details: https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md#pipelines'); using manual extractive QA
pytorch_model.bin: downloading bytes:           |  0.00B
model.safetensors: downloading bytes:           |  0.00B
QA model loaded!
```

<details><summary><code># class RAGAssistant:</code></summary>

```python
class RAGAssistant:
    """
    A complete RAG-powered knowledge assistant.

    Your implementation must include:
    1. Document retrieval from ChromaDB
    2. Answer generation using the QA model
    3. Confidence-based response handling
    4. Source attribution
    """

    def __init__(self, collection, qa_model, n_results: int = 8, confidence_threshold: float = 0.3):
        """
        Initialize the RAG assistant.

        Args:
            collection: ChromaDB collection with embedded documents
            qa_model: Hugging Face QA pipeline
            confidence_threshold: Minimum confidence to provide a direct answer
        """
        self.collection = collection
        self.qa_model = qa_model
        self.confidence_threshold = confidence_threshold
        self.n_results = n_results

    def retrieve(self, query: str, n_results: int = 8) -> Dict:
        """
        Retrieve relevant documents for a query.

        Args:
            query: The user's question
            n_results: Number of documents to retrieve

        Returns:
            Dict containing documents, metadatas, and ids
        """

        results = self.collection.query(
            query_texts=[query],
            n_results = self.n_results
        )

        return results

    def generate_answer(self, question: str, context: str) -> Dict:
        """
        Generate an answer from the given context.

        Args:
            question: The user's question
            context: Retrieved context to answer from

        Returns:
            Dict with answer and confidence score
        """

        result = self.qa_model(question=question,
                               context=context)

        return {
            'answer':result['answer'],
            'confidence':result['score']
        }

    def ask(self, question: str, n_results:int = 8) -> Dict:
        """
        Complete RAG pipeline: retrieve and generate answer.

        This method should:
        1. Retrieve relevant documents
        2. Combine them into context
        3. Generate an answer
        4. Handle low confidence appropriately
        5. Include source attribution

        Args:
            question: The user's question
            n_results: Number of documents to retrieve

        Returns:
            Dict with:
            - question: Original question
            - answer: Generated answer (or fallback message)
            - confidence: Confidence score
            - sources: List of source documents used
            - is_confident: Boolean indicating if confidence met threshold
        """

        retrieved = self.retrieve(query=question,
                                  n_results = self.n_results)

        #context = " ".join(retrieved['documents'][0])

        top_chunks = retrieved['documents'][0][:2] #choose top 5 articles
        context = "\n\n".join(top_chunks)

        answer_result = self.generate_answer(question=question,
                                             context=context)

        sources = [meta['title'] for meta in retrieved['metadatas'][0]]
        # print({
        #     'question': question,
        #     'answer': answer_result['answer'],
        #     'confidence': answer_result['confidence'],
        #     'sources': sources,
        #     'context': context
        # })
        return {
            'question': question,
            'answer': answer_result['answer'],
            'confidence': answer_result['confidence'],
            'is_confident': answer_result['confidence'] > self.confidence_threshold,
            'sources': sources,
            'context': context
        }

    def format_response(self, result: Dict) -> str:
        """
        Format the result into a user-friendly response string.

        Args:
            result: Output from the ask() method

        Returns:
            Formatted string response
        """

        question = result['question']

        #print(f"Question: {question}")
        #print("="*50)

        #print(f"\n" + "="*50)
        print(f"\n✅ Answer: {result['answer']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"   Sources: {', '.join(set(result['sources']))}")
        return ""


# Create your assistant
assistant = RAGAssistant(collection, qa_model, n_results=30, confidence_threshold=0.5)

print("✅ RAG Assistant created!")
```

</details>

```text
✅ RAG Assistant created!
```

<details><summary><code>test_questions = [</code></summary>

```python
test_questions = [
    "What are the calorie needs for a patient with Acute Respiratory Distress Syndrome (ARDS)?",
    "Is the routine use of disease specific formulas (i.e. diabetic) indicated in critical care?",
    "When should PN we initiated for a patient in unable to tolerate tube feeds (i.e. bowel discontinuity)?",
    'What is the standard calorie provision for the ICU patient?'
]

print("INITIAL TESTING")
print("=" * 60)
for question in test_questions:
    result = assistant.ask(question)
    print(f"\nQ: {question}")
    print(assistant.format_response(result))
```

</details>

```text
INITIAL TESTING
============================================================

Q: What are the calorie needs for a patient with Acute Respiratory Distress Syndrome (ARDS)?

✅ Answer: 20–35 kcal/kg/d
   Confidence: 51.56%
   Sources: [ASPEN_2016_guidelines][N. Sepsis], [ASPEN_2016_guidelines][H. When Indicated, Maximize Efficacy of PN], [ASPEN_2016_guidelines][I. Pulmonary Failure], [ASPEN_2016_guidelines][J. Renal Failure], [ASPEN_2016_guidelines][C. Dosing of EN], [ASPEN_2016_guidelines][Open Abdomen], [ASPEN_2016_guidelines][Burns], [ASPEN_2016_guidelines][K. Hepatic Failure], [ASPEN_2016_guidelines][A. Nutrition Assessment], [ASPEN_2016_guidelines][Q. Obesity in Critical Illness], [ASPEN_2016_guidelines][Trauma]


Q: Is the routine use of disease specific formulas (i.e. diabetic) indicated in critical care?

✅ Answer: Disease-specific and severe fluid-restricted formulas may be rarely used in a small percentage of patients on a case-by-case basis
   Confidence: 8.35%
   Sources: [ASPEN_2016_guidelines][H. When Indicated, Maximize Efficacy of PN], [ASPEN_2016_guidelines][F. Adjunctive Therapy], [ASPEN_2016_guidelines][N. Sepsis], [ASPEN_2016_guidelines][E. Selection of Appropriate Enteral Formulation], [ASPEN_2016_guidelines][J. Renal Failure], [ASPEN_2016_guidelines][C. Dosing of EN], [ASPEN_2016_guidelines][D. Monitoring Tolerance and Adequacy of EN], [ASPEN_2016_guidelines][G. When to Use PN], [ASPEN_2016_guidelines][K. Hepatic Failure], [ASPEN_2016_guidelines][A. Nutrition Assessment], [ASPEN_2016_guidelines][Q. Obesity in Critical Illness]


Q: When should PN we initiated for a patient in unable to tolerate tube feeds (i.e. bowel discontinuity)?

✅ Answer: over the first 7 days following ICU admission
   Confidence: 10.32%
   Sources: [ASPEN_2016_guidelines][H. When Indicated, Maximize Efficacy of PN], [ASPEN_2016_guidelines][B. Initiate EN], [ASPEN_2016_guidelines][C. Dosing of EN], [ASPEN_2016_guidelines][O. Postoperative Major Surgery (SICU Admission Expected)], [ASPEN_2016_guidelines][Open Abdomen], [ASPEN_2016_guidelines][D. Monitoring Tolerance and Adequacy of EN], [ASPEN_2016_guidelines][G. When to Use PN], [ASPEN_2016_guidelines][Burns], [ASPEN_2016_guidelines][K. Hepatic Failure], [ASPEN_2016_guidelines][Q. Obesity in Critical Illness], [ASPEN_2016_guidelines][L. Acute Pancreatitis]


Q: What is the standard calorie provision for the ICU patient?

✅ Answer: 25 kcal/kg/d
   Confidence: 47.09%
   Sources: [ASPEN_2016_guidelines][H. When Indicated, Maximize Efficacy of PN], [ASPEN_2016_guidelines][N. Sepsis], [ASPEN_2016_guidelines][E. Selection of Appropriate Enteral Formulation], [ASPEN_2016_guidelines][I. Pulmonary Failure], [ASPEN_2016_guidelines][J. Renal Failure], [ASPEN_2016_guidelines][C. Dosing of EN], [ASPEN_2016_guidelines][Burns], [ASPEN_2016_guidelines][A. Nutrition Assessment], [ASPEN_2016_guidelines][Q. Obesity in Critical Illness], [ASPEN_2016_guidelines][Trauma]
```

---

## Part 7: Comprehensive Testing

Test your system thoroughly with at least 15 questions covering different types and difficulty levels.

<details><summary><code>test_suite = {</code></summary>

```python

test_suite = {
    "simple_facts": [
        # Questions with straightforward, single-fact answers
        # Example: "What is the price of X?" "When does Y open?"
        "What is the standard calorie range for the ICU patient?",
        "What is the standard protein range for the ICU patient?",
        "What is the maximum safe protein provision for the ICU patient?",
        "What is the best method for determining energy needs in the critically ill adult patient?",
        "Are the bowel sounds required before the initation of enteral tube feeds?",
    ],

    "detailed_explanations": [
        # Questions requiring more detailed answers
        # Example: "How does X work?" "What are the features of Y?"
        "What considerations should be made for enteral feeds when a patient is on pressors?",
        "What are the contraindications against the initation of parenteral nutrition?",
        "How do you mitigate risk of fungemia in parenteral nutrition?",
        "What is the advantage of using standardized commercially available PN (premixed PN) versus compounded PN admixtures?",
        "Explain the progression of acute phase sepsis and shock in the ICU patient.",
    ],

    "comparison_or_complex": [
        # More complex questions
        # Example: "What's the difference between X and Y?"
        "What are the protein requirements for a burn patient versus that of a standard ICU patient?",
        "What is the optimal timing for initiating supplemental PN when EN does not meet energy or protein goals in the patient at high or low nutrition risk?",
    ],

    "edge_cases": [
        # Questions NOT answerable from your knowledge base
        # These test your fallback handling
        "What are the calorie needs a patient with both severe burns and is morbidly obese?",
        "When should supplemental PN be discontinued?",
        "When can intravenous lipid emulsions (IVFE) be safely started for a patient with fungemia?",
    ]
}
```

</details>

<details><summary><code># Run comprehensive tests and collect results</code></summary>

```python
# ============================================================
# Run comprehensive tests and collect results
# ============================================================

test_results = []

print("COMPREHENSIVE TEST RESULTS")
print("=" * 70)

for category, questions in test_suite.items():
    print(f"\n\n{'='*70}")
    print(f"CATEGORY: {category.upper().replace('_', ' ')}")
    print("=" * 70)

    for question in questions:
        result = assistant.ask(question)

        # Store result for analysis
        test_results.append({
            "category": category,
            "question": question,
            "answer": result["answer"],
            "confidence": result["confidence"],
            "is_confident": result["is_confident"],
            "sources": result["sources"]
        })

        # Display result
        confidence_indicator = "✅" if result["is_confident"] else "⚠️"
        print(f"\n{confidence_indicator} Q: {question}")
        print(f"   A: {result['answer']}")
        print(f"   Confidence: {result['confidence']:.2%} | Sources: {', '.join(result['sources'][:2])}")
```

</details>

```text
COMPREHENSIVE TEST RESULTS
======================================================================


======================================================================
CATEGORY: SIMPLE FACTS
======================================================================

⚠️ Q: What is the standard calorie range for the ICU patient?
   A: 25 kcal/kg/d
   Confidence: 12.68% | Sources: [ASPEN_2016_guidelines][Q. Obesity in Critical Illness], [ASPEN_2016_guidelines][A. Nutrition Assessment]

✅ Q: What is the standard protein range for the ICU patient?
   A: 1.2–2.0 g/kg actual body weight per day
   Confidence: 76.79% | Sources: [ASPEN_2016_guidelines][C. Dosing of EN], [ASPEN_2016_guidelines][C. Dosing of EN]

⚠️ Q: What is the maximum safe protein provision for the ICU patient?
   A: 1.2–2.0 g/kg actual body weight per day
   Confidence: 21.83% | Sources: [ASPEN_2016_guidelines][C. Dosing of EN], [ASPEN_2016_guidelines][C. Dosing of EN]

⚠️ Q: What is the best method for determining energy needs in the critically ill adult patient?
   A: indirect calorimetry
   Confidence: 47.77% | Sources: [ASPEN_2016_guidelines][A. Nutrition Assessment], [ASPEN_2016_guidelines][Burns]

⚠️ Q: Are the bowel sounds required before the initation of enteral tube feeds?
   A: overt signs of contractility should not be required
   Confidence: 34.32% | Sources: [ASPEN_2016_guidelines][B. Initiate EN], [ASPEN_2016_guidelines][B. Initiate EN]


======================================================================
CATEGORY: DETAILED EXPLANATIONS
======================================================================

⚠️ Q: What considerations should be made for enteral feeds when a patient is on pressors?
   A: close monitoring for signs and symptoms of intolerance
   Confidence: 29.42% | Sources: [ASPEN_2016_guidelines][D. Monitoring Tolerance and Adequacy of EN], [ASPEN_2016_guidelines][O. Postoperative Major Surgery (SICU Admission Expected)]

⚠️ Q: What are the contraindications against the initation of parenteral nutrition?
   A: Placement and maintenance of enteral access devices in patients who cannot maintain volitional intake have potential complications
   Confidence: 29.90% | Sources: [ASPEN_2016_guidelines][C. Dosing of EN], [ASPEN_2016_guidelines][H. When Indicated, Maximize Efficacy of PN]

⚠️ Q: How do you mitigate risk of fungemia in parenteral nutrition?
   A: the amount of energy delivered by the parenteral route should be reduced appropriately to compensate for the increase in the energy being delivered enterally
   Confidence: 17.14% | Sources: [ASPEN_2016_guidelines][H. When Indicated, Maximize Efficacy of PN], [ASPEN_2016_guidelines][C. Dosing of EN]

✅ Q: What is the advantage of using standardized commercially available PN (premixed PN) versus compounded PN admixtures?
   A: improved safety
   Confidence: 94.45% | Sources: [ASPEN_2016_guidelines][H. When Indicated, Maximize Efficacy of PN], [ASPEN_2016_guidelines][H. When Indicated, Maximize Efficacy of PN]

⚠️ Q: Explain the progression of acute phase sepsis and shock in the ICU patient.
   A: The combination of compromised GI function and hypermetabolism from an exaggerated acute phase response likely leads to greater risk for malnutrition in this subpopulation of critically ill patients
   Confidence: 5.24% | Sources: [ASPEN_2016_guidelines][N. Sepsis], [ASPEN_2016_guidelines][N. Sepsis]


======================================================================
CATEGORY: COMPARISON OR COMPLEX
======================================================================

⚠️ Q: What are the protein requirements for a burn patient versus that of a standard ICU patient?
   A: 1.4 g/kg/d versus 2.2 g/kg/d
... (22 more lines)
```

<details><summary><code># Analyze test results</code></summary>

```python
# ============================================================
# Analyze test results
# ============================================================

print("\n" + "=" * 70)
print("TEST RESULTS SUMMARY")
print("=" * 70)

total_questions = len(test_results)
confident_answers = sum(1 for r in test_results if r["is_confident"])
avg_confidence = sum(r["confidence"] for r in test_results) / total_questions

print(f"\nTotal questions tested: {total_questions}")
print(f"Confident answers: {confident_answers} ({confident_answers/total_questions:.1%})")
print(f"Low confidence answers: {total_questions - confident_answers}")
print(f"Average confidence: {avg_confidence:.2%}")

# By category
print("\nResults by category:")
for category in test_suite.keys():
    category_results = [r for r in test_results if r["category"] == category]
    cat_confident = sum(1 for r in category_results if r["is_confident"])
    cat_avg = sum(r["confidence"] for r in category_results) / len(category_results)
    print(f"  {category}: {cat_confident}/{len(category_results)} confident, avg confidence: {cat_avg:.2%}")
```

</details>

```text

======================================================================
TEST RESULTS SUMMARY
======================================================================

Total questions tested: 15
Confident answers: 2 (13.3%)
Low confidence answers: 13
Average confidence: 30.42%

Results by category:
  simple_facts: 1/5 confident, avg confidence: 38.68%
  detailed_explanations: 1/5 confident, avg confidence: 35.23%
  comparison_or_complex: 0/2 confident, avg confidence: 20.92%
  edge_cases: 0/3 confident, avg confidence: 14.98%
```

---

## Part 8: Analysis

### 8.1 Performance Analysis

Our RAG system was able to handle simple facts (i.e. What is the standard protein range for the ICU patient? -> 1.2 g to 2.0 g/kg). General medical guidance and straight figures can be found in sigular documents and in clear, direct sentences.  

Our system still struggles with cases that deviated from baseline patients (i.e. bowel discontinuity or obesity). This becuase our system was not able to recognize that questions were specifically looking for cases that deviated from "normal" patients. Logic on clincal guidlines for these cases also spanned multiple documents.

### 8.3 Future Improvements

-----

In document preparation we used an open source libary `pymupdf4llm` to first convert a PDF into markdown. We then relied on text/characters at land marks to split the research paper into different documents. For a cleaner knowledge base we can manually convert the PDF into plain text, allowing us tighter control over how the knowledge base is structured. More intentional hierarchical chunking can also be done to make sure the scope/context of each chunk is limited to specific clinical scenarios.

We can use query expasion/generation so that our queries more closely match our embeddings nad vectors. Likewise, for questions that require synthesis of data that spread across multiple documnets was can use generative models such as`google/flan-t5-base` to provide more comprehenisve answers. At 250 MB, this model would still be relatively light weight and runnable on local CPUs.

---

## Part 9: Demo Showcase

<details><summary><code>def run_demo(assistant, demo_questions: List[str]):</code></summary>

```python
def run_demo(assistant, demo_questions: List[str]):
    """
    Run a polished demo of the RAG assistant.
    """
    print("\n" + "="*70)
    print(f"{PROJECT_INFO['topic']} - Knowledge Assistant Demo")
    print(f"   {PROJECT_INFO['description']}")
    print("="*70)

    for i, question in enumerate(demo_questions, 1):
        print(f"\n{'─'*70}")
        print(f"User Question {i}:")
        print(f"   \"{question}\"")
        print()

        result = assistant.ask(question)

        print(f"Assistant Response:")
        print(f"   {result['answer']}")
        print()
        print(f"   Sources: {', '.join(result['sources'][:2])}")
        print(f"   Confidence: {result['confidence']:.1%}")

    print(f"\n{'='*70}")
    print("Demo complete!")
    print("="*70)


demo_questions = [
    "What is the standard protein range for the ICU patient?",
    "What is the gold-standard for determining energy needs in the ICU?",
    "Are the bowel sounds required before the initation of enteral tube feeds",
    "Explain the progression of the acute phase repsonse and shock in the ICU patient?",
    "When should parenteral nutrition be initatied?",
]

run_demo(assistant, demo_questions)
```

</details>

```text

======================================================================
Nutrition Intervention in the ICU - Knowledge Assistant Demo
   This RAG LLM is trained with ASPENs 2016 and 2021 practice guidelines for nutrition intervention in the ICU.
======================================================================

──────────────────────────────────────────────────────────────────────
User Question 1:
   "What is the standard protein range for the ICU patient?"

Assistant Response:
   1.2–2.0 g/kg actual body weight per day

   Sources: [ASPEN_2016_guidelines][C. Dosing of EN], [ASPEN_2016_guidelines][C. Dosing of EN]
   Confidence: 76.8%

──────────────────────────────────────────────────────────────────────
User Question 2:
   "What is the gold-standard for determining energy needs in the ICU?"

Assistant Response:
   indirect calorimetry

   Sources: [ASPEN_2016_guidelines][A. Nutrition Assessment], [ASPEN_2016_guidelines][A. Nutrition Assessment]
   Confidence: 80.3%

──────────────────────────────────────────────────────────────────────
User Question 3:
   "Are the bowel sounds required before the initation of enteral tube feeds"

Assistant Response:
   overt signs of contractility should not be required

   Sources: [ASPEN_2016_guidelines][B. Initiate EN], [ASPEN_2016_guidelines][B. Initiate EN]
   Confidence: 30.6%

──────────────────────────────────────────────────────────────────────
User Question 4:
   "Explain the progression of the acute phase repsonse and shock in the ICU patient?"

Assistant Response:
   exaggerated acute phase response likely leads to greater risk for malnutrition

   Sources: [ASPEN_2016_guidelines][N. Sepsis], [ASPEN_2016_guidelines][B. Initiate EN]
   Confidence: 6.3%

──────────────────────────────────────────────────────────────────────
User Question 5:
   "When should parenteral nutrition be initatied?"

Assistant Response:
   the first week of hospitalization

   Sources: [ASPEN_2016_guidelines][G. When to Use PN], [ASPEN_2016_guidelines][C. Dosing of EN]
   Confidence: 13.8%

======================================================================
Demo complete!
======================================================================
```
