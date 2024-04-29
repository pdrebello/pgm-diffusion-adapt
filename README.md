# Diffusion Adapt
A 10-708: Probablistic Graphical Models course project to leverage diffusion models for the task of domain adaption

Real Images             |  Synthetic Images (Generated without Access to True Labels)
:-------------------------:|:-------------------------:
![dadapt_real](https://github.com/pdrebello/pgm-diffusion-adapt/assets/46563240/97527693-aa2d-46fe-88da-ded06eb940da) |  ![dadapt](https://github.com/pdrebello/pgm-diffusion-adapt/assets/46563240/a2f69414-4248-4997-92f9-582ec522f88f)






### Steps to Run ###

### Data

1. Download file from 
2. Unzip (replace existing data directory if asked)
    ```sh
    unzip data.zip
    ```
### Running Baselines

1. For PoolNN baseline run
    ```sh
     python baseline.py
     ```
    This trains a CNN on images from the three source domains, and tests on images from the target domain

2. For the gold standard baseline run
    ```sh
     python baseline_gold.py
     ```
    This assumes we have access to gold training labels for the target domain and trains a CNN on them. It gives us a goal accuracy for domain adaptation (an upper bound)

### Running Diffusion Adapt

1. Run
    ```sh
     python alternating_diffusion.py
     ```
  This will run 10 cycles of alternating diffusion as follows:
  - (i) Train a classifier with poolNN (baseline)
  
  - (ii) Train a diffusion model using source data (with gold labels) and target data (with pseudo labels from trained classifier)
  
  - (iii) Use diffusion model to generate synthetic images from the target domain
  
  - (iv) Use synthetic images to further train the classifier and return to (ii)
