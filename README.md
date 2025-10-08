# AI Driven Early Warning Systems for Marine Stingers (computer vision models) 

Bluebottles frequently appear on Australian beaches, causing painful stings and posing a public safety challenge. Despite their lack of swimming ability, these organisms are transported by ocean currents, wind patterns, and wave activity, leading to variable beaching events. The objective of this project is to develop an AI model linking bluebottle occurrences along eastern coast of Australia to environmental factors such as wind and waves, enabling real time forecasting of bluebottle presence, and integrating some temporal continuity. 
By incorporating environmental drivers and coastal characteristics, we will assess the relative importance of these factors using advanced computer vision models. Beyond single point analysis, the project also considers neighbouring ocean areas adjacent to the coastline. By analysing spatial patches offshore and applying computer vision techniques, we aim to capture broader spatial patterns of environmental change such as shifts in wind, wave, and current fields that may influence bluebottle transport.  
This patch based approach enhances the model’s ability to detect spatiotemporal patterns of daily variability, providing more detailed understanding of how environmental dynamics drive beaching events. 





## 🧰 Requirements
- It is recommended to use **VS Code** or **JupyterLab** 
- **[Anaconda](https://www.anaconda.com/products/distribution) / Miniconda** installed on your system 
- **Python version:** ≥ 3.10.18  
- **Operating System:** Windows, macOS, or Linux

---

## ⚙️ Environment Setup

1. **Clone or download the repository**  

    ```bash
    git clone <repo_url>
    cd <repo_name>
    ```

2. **Create a new Conda environment** using the provided YAML file:

   ```bash
   conda env create -f bluebottle.yaml
   ```

3. **Activate the environment**:

   ```bash
   conda activate bluebottle
   ```

4. **Verify the Python version** (should be ≥ 3.10.18):

   ```bash
   python --version
   ```

---

## 🚀 Running the Notebook

1. **Launch Jupyter Notebook**:

   ```bash
   jupyter notebook
   ```

2. In the Jupyter interface, **open** the file:

   ```
   EDA.ipynb
   ```

3. **Run all cells** sequentially to execute the analysis.

---

## 🧩 Troubleshooting

* If the environment name in `bluebottle.yaml` conflicts with an existing one, rename it in the file before creating:

  ```yaml
  name: bluebottle
  ```
* To update dependencies later:

  ```bash
  conda env update -f bluebottle.yaml --prune
  ```
* If Jupyter does not detect the new environment:

  ```bash
  python -m ipykernel install --user --name=bluebottle --display-name "Python (bluebottle)"
  ```

---

## 📄 File Structure

```
│
├── output              # Output folder
├── EDA.ipynb           # Jupyter notebook containing the EDA process
├── bluebottle.yaml     # Conda environment file
└── README.md           # Setup and usage instructions
```

---

## 🧠 Notes

* Always activate the environment before running the notebook.



---

