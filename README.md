# Biomedical_NER_and_Relation_Extraction

A biomedical NLP pipeline built using PubMedBERT for Chemical/Disease Named Entity Recognition (NER) and Chemical–Disease Relation Extraction on the BC5CDR dataset.

## Features

- Chemical and Disease entity recognition
- CID relation extraction
- Interactive knowledge graph visualization
- Confidence scoring and annotated outputs
- Streamlit-based dashboard

## Tech Stack
<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Transformers-FFBF00?style=for-the-badge&logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/PubMedBERT-0A66C2?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/NetworkX-000000?style=for-the-badge" />
</p>

## Dataset

- BC5CDR (BioCreative V Chemical Disease Relation Dataset)
- 95,000+ labeled biomedical tokens
- 
## Interface
![image](Assets/2.png)
![image](Assets/1.png)

## Example

### Input

```text
Cisplatin-induced nephrotoxicity is a major dose-limiting side effect.
```

### Output

- Chemical: Cisplatin
- Disease: nephrotoxicity
- Relation: Cisplatin → CID → nephrotoxicity

## Author

Nikita Maurya
