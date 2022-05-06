# Preparing for Ablated R2R/RxR-en/Touchdown Instructions

## Download Processed Instructions
We provide the processed intructions for R2R/RxR-en/Touchdown, which can be downloaded by the following script:

```bash
cd Diagnose_VLN/
python data_processing/download_data.py --download_instructions
```



## Generate Instructions from Scratch

We also provide the script to process the instructions from scratch. First download the raw instructions by:

```bash
cd Diagnose_VLN/
python data_processing/download_data.py --download_raw_instructions
```

Then refer to [process_indoor_instructions.ipynb](process_indoor_instructions.ipynb) to create instruction ablations for R2R or RxR-en, or [process_outdoor_instructions.ipynb](process_outdoor_instructions.ipynb) to process Touchdown instructions.
We use [Stanza](https://stanfordnlp.github.io/stanza/) part-of-speech tagger in preprocessing.