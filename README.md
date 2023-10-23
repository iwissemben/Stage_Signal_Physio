
# Stage Signal Physiologique 
<img width="2300" alt="final_banner@2x" src="https://github.com/iwissemben/custom-signal-toolbox-art/blob/main/illustrations/imgs/Output/2x/final_banner@2x.png?raw=true">

## Description 

This repository is dedicated to my end of master internship. The aim of the internship is to learn to collect and manipulate multimodal physiological signals and data obtained with various techniques such as EEG, fNIRS and motion-tracking. 

For this project I joined the ReArm research team, and their work serve as a guide to my project.

The first phase of the project is to analyze EEG signal. I hope to work on fNIRS and motion-tracking data next. 

## Table of contents
* [Description](#Description)
* [Installation](#Installation)
* [Structure](#Structure)
* [FAQ](#FAQ)

## Installation

1. Import the project using GitHub and GitHub desktop application by cloning the repository to your computer.
 - [x]  In the project folder check that you can find the `Stage_Signal_Physio/HELP/environnement` folder and the `wissem_venv.yml` file inside.
2. For the first run, replicate the environement using terminal:
   - **List** all your conda environnements : `conda info --envs`
     - If the `wissem_venv` environnement **is listed**, it is already installed.
     - If you want to delete it: `conda remove --name wissem_venv --all`
     - If `wissem_venv` **is not listed**, create it (2.2).
   - **Create** the environnement from yaml file : `conda env create -f ./HELP/environnement/wissem_venv.yml`
   - **Verify creation** of `wissem_venv` environnement : `conda info --envs`
3. **Activate** wissem_venv : `conda activate wissem_venv`
   - **Verify activation** of wissem_venv : `conda info --envs`. Active env should be highlighted like following: `wissem_venv*`
5. 🎉You are all set! 🥳
- Run the script you want (`EEG_script.py` for instance) with your favorite code editor in the right environment.

```python
  x="Hello World"
  print("Hello World")
```
Screencaps of the steps :
| STEP              | Windows                                                         | macOS                                                                                                        |
|:-----------------:|:---------------------------------------------------------------:|:---------------------------------------------------------------:|
| 2.1 - 2.2         | [1](HELP/screencaps/windows/windows_conda_create_env_f.PNG)     |[A](HELP/screencaps/macos/macos_conda_create_env.png)|
| 2.3 - 3           | [2](HELP/screencaps/windows/windows_conda_created_env.PNG)      |[B1](HELP/screencaps/macos/macos_conda_created_env.png) [B2](HELP/screencaps/macos/macos_conda_loaded_env.png)|
| 4                 | [3](HELP/screencaps/windows/windows_conda_loading_worked.PNG)   |[C](HELP/screencaps/macos/macos_loading_worked.png) |

## Structure 

The project Stage_Signal_Physio is organized as a directory with the following structure:

### Root directory
    ├── Stage_Signal_Physio     # Project root directory
    │   ├── .vscode             # VSCode workspace settings
    │   ├── DAT                 # Contains input and output DATA
    │   ├── HELP                # Contains helping files
    │   ├── SRC                 # Scripts and notebooks
    │   └── README.md           

### .vscode
    ├── .vscode                 # VSCode workspace settings
    │   ├── launch.json         # Settings to configure VSCode's debugger
    │   └── settings.json       # Settings to configure VSCode
### DAT
    ├── DAT                     # Contains input and output DATA
    │   ├── INPUT               # Contains the input data (datasets)
    │   └── OUTPUT              # Contains the output data (datasets,plots as pictures)

### HELP
    ├── HELP                                               # Contains helping files
    │   ├── 20201112153755_001_MoLu_c_eeg.info             # Recording settings
    │   ├── EEG_Analysis_FunctionalTasks_Details.docx      # Details of test protocol
    │   └── settings note.txt                              # Troubleshooting

### SRC
    ├── SRC                                   # Scripts and notebooks
    │   ├── _pycache_                         # Compiled scripts
    │   ├── matlab_scripts                    # Contains Matlab scripts
    │   ├── EEG_script.py                     # Main script EEG for EEG analysis
    │   ├── my_filters.py                     # Script containing custom functions related to filtering
    │   ├── my_functions.py                   # Script containing general custom functions
    │   ├── comparisons_1channel1000Hz.ipynb  # Jupyter Notebook Ramdani signal comparisons matlab
    │   ├── comparisons_Rearm.ipynb           # Jupyter Notebook rearm data PSDs comparisons
    │   ├── export_signals.ipynb              # Jupyter Notebook  for exporting xdf EEG data
    │   ├── filters.ipynb                     # Jupyter Notebook for testing filters
    │   └── sandbox.ipynb                     # Jupyter Notebook as sandbox

This folder contains the scripts to run. Each is dedicated to study a subject. The main script is 'EEG_script.py". 

- 'EEG_script.py' it reads the raw data (.xdf file in DAT/INPUT), and processes the EEG signal to obtain the ERSP and represents each electrode's signal in a time frequency plot (cf. [Kosei Nakayashiki publication](https://pubmed.ncbi.nlm.nih.gov/24886610/)).
- 'my_functions.py' and 'my_filters.py' are scripts that contain custom functions. These are like tailor-made toolboxes that are called in other scripts such as 'EEG_script.py'.



## FAQ
### What is the aim of the EEG part of the internship?
- The aim is to learn and use signal processing methods (time-frequency analysis) and tools to extract Event Related Spectral Perturbations (ERSP) from a recording during a particular event (or task). The hypothesis is that the event will generate task specific ERSP, and enable us to gain insight on the electrophysiological effects of the rehabilitation program on patients.
### Where should I start?
- After installing the project, you can start by running the EEG_script.py file. After all it is the main script for analyzing EEG signals in this project.
- The other scripts are here for experimenting and testing separately different parameters.
### Do I need the Visual Studio Code editor (VSCode) to run the script?
- Not necessarily. Actually, I started the project by coding with Spyder's IDE. I am taking advantage of this internship to learn many new programming tools and methods here (starting with GitHub, VSCode and CONDA environements).
- You are free to run the scripts with your favorite code editor as long as you are used to it, you'll manage settings specific to your workstation.

### I don't find the latest changes, where should i look?
-For this project, I am working on different GitHub branches. The main branch is for approved changes, other branches are organized for each "Assignment" or objective. Once the objective(s) is (are) met, I make a pull request and if there is no conflict with the main branch, I create a merge request to merge it to the main branch.

-Sometimes i get carried away by changes i find necessary, so a branch may contain lots of changes, so you may want to take a look to them.

## Feedback

If you have any suggestions, or discover information that might be relevant to this project, feel free to edit the FAQ in the 'readme.md' file or open a discussion thread to share it with everyone.

You can always make Pull Requests to alert other contributors of changes and pique their curiosity.
