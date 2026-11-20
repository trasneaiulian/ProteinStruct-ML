# ![CI logo](https://codeinstitute.s3.amazonaws.com/fullstack/ci_logo_small.png)

# ProteinStruct-ML: Predicting Protein Structural Deviation Using Machine Learning

## Project Overview

Proteins are biological macromolecules involved in nearly every process occurring within living cells. Their functions depend not only on their amino acid sequences but also on the three-dimensional structures they adopt.

Understanding protein structure is therefore essential for investigating biological mechanisms, interpreting protein function, and supporting research in biotechnology, molecular biology, and drug discovery.

Experimentally determining protein structures can require substantial time and resources. Computational approaches provide an alternative by generating structural models, but these models can vary considerably in their accuracy.

Assessing the quality of predicted protein structures is consequently an important problem in structural bioinformatics.

This project investigates whether machine-learning algorithms can predict protein structural deviation using numerical descriptors associated with computational protein models.

The analysis uses the **CASP dataset**, which contains nine numerical input features (F1–F9) and a target variable representing structural deviation, expressed as Root Mean Square Deviation (RMSD).

The project follows a complete data-science workflow, including:

- data extraction, cleaning, and validation;
- exploratory data analysis and statistical testing;
- development and comparison of regression models;
- hyperparameter optimization and cross-validation;
- evaluation of predictive performance;
- and deployment of an interactive application using Streamlit and Heroku.

The final application allows users to explore the dataset, review model performance, and generate RMSD predictions from numerical input features.

## Understanding the Biological Context

### Protein Structure

Proteins consist of amino acids connected into polypeptide chains. These chains fold into three-dimensional structures that influence how proteins interact with other molecules and perform their biological functions.

Protein structure is commonly described at several organizational levels:

- **Primary structure:** The amino acid sequence.
- **Secondary structure:** Local structural arrangements such as alpha helices and beta sheets.
- **Tertiary structure:** The overall three-dimensional organization of a polypeptide chain.
- **Quaternary structure:** The arrangement of multiple polypeptide chains within a protein complex.

The three-dimensional organization of a protein is particularly important because changes in structural conformation can influence its biological activity.

### Experimental and Computational Protein Structures

Protein structures can be determined experimentally using techniques such as X-ray crystallography, nuclear magnetic resonance (NMR) spectroscopy, and cryogenic electron microscopy (cryo-EM).

Computational methods can also generate structural models based on amino acid sequences and other available information.

However, computational models are not necessarily identical to experimentally determined reference structures.

Evaluating the differences between predicted and reference structures is therefore important when assessing structural model quality.

### Root Mean Square Deviation (RMSD)

Root Mean Square Deviation is a commonly used measure of structural difference between corresponding atoms in two molecular structures.

After appropriate structural alignment, RMSD can be expressed as:

\[
\mathrm{RMSD} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}\left\|\mathbf{r}_i-\mathbf{r}_i^{\mathrm{ref}}\right\|^2}
\]

where:

- \(N\) is the number of corresponding atoms;
- \(\mathbf{r}_i\) is the position of atom \(i\) in the evaluated structure;
- \(\mathbf{r}_i^{\mathrm{ref}}\) is the corresponding position in the reference structure.

RMSD is commonly expressed in ångströms (Å), where 1 Å equals 0.1 nanometres.

A smaller RMSD generally indicates greater structural similarity between the compared atomic coordinates, whereas a larger RMSD indicates greater structural deviation.

RMSD depends on the atoms selected, their correspondence, and the alignment procedure. It should therefore be interpreted in the context of how the structures were compared.

### Machine Learning for Structural Deviation Prediction

Machine learning provides methods for identifying statistical relationships between input variables and an outcome of interest.

In supervised regression, an algorithm learns from observations for which both the input features and the target values are known.

After training, the model can estimate the target variable for new observations.

In ProteinStruct-ML, the input consists of nine numerical descriptors (F1–F9), while RMSD is the continuous target variable.

The central question is whether these descriptors contain sufficient predictive information to estimate structural deviation with useful accuracy.

This project compares several regression approaches and evaluates their performance on data not used during model training.

## Project Objectives

The overall objective of ProteinStruct-ML is to investigate whether machine-learning algorithms can accurately estimate protein structural deviation (RMSD) using numerical descriptors from the CASP dataset.

The project combines exploratory statistical analysis with supervised machine learning to evaluate relationships between structural descriptors and RMSD.

The specific objectives are to:

1. **Prepare a reliable dataset:** Inspect the original CASP dataset for missing values, duplicate records, inconsistent data types, and potentially unusual observations.

2. **Explore structural descriptors:** Investigate the distributions of RMSD and the nine numerical features (F1–F9), identify extreme observations, and examine relationships between variables.

3. **Evaluate statistical relationships:** Use correlation analysis and hypothesis testing to determine whether selected descriptors are statistically associated with RMSD.

4. **Develop regression models:** Establish a baseline prediction using DummyRegressor and compare the performance of Linear Regression, Random Forest Regression, and HistGradientBoosting Regression.

5. **Optimize model performance:** Apply cross-validation and hyperparameter optimization to identify suitable model configurations.









Welcome,

This is the Code Institute student template for the three Data Analytics capstone projects. We have preinstalled all of the tools you need to get started. It's perfectly okay to use this template as the basis for your project submissions. Click the `Use this template` button above to get started.

You can safely delete the Template Instructions section of this README.md file and modify the remaining paragraphs for your own project. Please do read the Template Instructions at least once, though! It contains some important information about the IDE and the extensions we use.

If you are working on the first capstone project, you can also delete `.python-version`, `.slugignore`, `Procfile` and `setup.sh` as they are only required for later dashboard projects. 

## How to use this repo

1. Use this template to create your GitHub project repo. Click the **Use this template** button, then click **Create a new repository**.

1. Copy the URL of your repository to your clipboard.

1. In VS Code, select **File** -> **Open Folder**.

1. Select your `vscode-projects` folder, then click the **Select Folder** button on Windows, or the **Open** button on Mac.

1. From the top menu in VS Code, select **Terminal** > **New Terminal** to open a new terminal.

1. In the terminal, type `git clone` followed by the URL of your GitHub repository. Then hit **Enter**. This command will download all the files in your GitHub repository into your vscode-projects folder.

1. In VS Code, select **File** > **Open Folder** again.

1. This time, navigate to and select the folder for the project you just downloaded. Then, click **Select Folder**.

1. A virtual environment is necessary when working with Python projects to ensure each project's dependencies are kept separate. You need to create your virtual environment, also called a venv, and then activate it whenever you return to your workspace.
Click the gear icon in the lower left-hand corner of the screen to open the Manage menu and select **Command Palette** to open the VS Code command palette.

1. In the command palette, type: *create environment* and select **Python: Create Environment…**

1. Choose **Venv** from the dropdown list.

1. Choose the Python version you installed earlier. Currently, we recommend Python 3.12.8

1. **DO NOT** click the box next to `requirements.txt`; you need to complete additional steps before installing your dependencies. Click **OK**.

1. You will see a `.venv` folder appear in the file explorer pane, indicating that the virtual environment has been created.

1. **Important**: Note that the `.venv` folder is in the `.gitignore` file so that Git won't track it.

1. Return to the terminal by clicking on the TERMINAL tab, or click on the **Terminal** menu and choose **New Terminal** if no terminal is currently open.

1. In the terminal, use the command below to install your dependencies. This may take several minutes.

 ```console
 pip3 install -r requirements.txt
 ```

1. Open the `jupyter_notebooks` directory, and click on the notebook you want to open.

1. Click the **Kernel** button, then choose **Python Environments**.

Note that the kernel says `Python 3.12.8` as it inherits from the venv, so it will be Python-3.12.8 if that is what is installed on your PC. To confirm this, you can use the command below in a notebook code cell.

```console
! python --version
```

## Deployment Reminders

* The `.python-version`, `.slugignore`, `Procfile` and `setup.sh` files are necessary only if you are deploying a Streamlit app to Heroku as part of your submission for units 2 and 3. 
* Set the `.python-version` Python version to a [Heroku-22](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack, currently supported version that most closely matches what you used in this project.
* The project can be deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the **Deploy** tab, select **GitHub** as the deployment method.
3. Select your repository name and click **Search**. Once it is found, click **Connect**.
4. Select the branch you want to deploy, then click **Deploy Branch**.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button **Open App** at the top of the page to access your App.
6. If the slug size is too large, then add large files not required for the app to the `.slugignore` file.
