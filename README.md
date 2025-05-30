# Natural language processing course: `Automatic generation of Slovenian traffic news for RTV Slovenija`

<!-- Please, organize README and the whole structure of the repository to be self-contained and reproducible. -->
## Project Overview
This project is part of the Natural language course on Faculty of Computer and Information Science of University of Ljubljana. The assignment was to automatically generate traffic news report that is read on radio every 30 minutes based on the traffic information available on promet.si. 

This repository includes the code for our prompt engineering and evaluation pipeline, including data preprocessing, model prompting, and automated evaluation.
Repository Structure

    /report/Submission3.pdf
    Final project report detailing methodologies, experiments, and outcomes.
    Prompthub

    /model_prompting/
    Scripts and configurations used for generating prompts and interacting with language models.
    GitHub+1Prompting Guide+1

    /prompt_engineering_evaluation/
    Tools and scripts for the automated evaluation of prompt effectiveness across different models.

    /data_preprocessing/
    Code responsible for analyzing and preprocessing the dataset located in the /RTVSlo/ directory.

HPC Deployment

The complete codebase and associated models are available on the SLING HPC system at:


/d/hpc/projects/onj_fri/jazbeci

Note: If you encounter an exit code 255 error on the HPC, be aware that this is a known issue with the system.
