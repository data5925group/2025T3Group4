# AI Driven Early Warning Systems for Marine Stingers (Computer Vision Models)

Bluebottles frequently appear on Australian beaches, causing painful stings and posing a public safety challenge. This project develops an AI model linking bluebottle occurrences along the eastern coast of Australia to environmental factors such as ocean currents, wind patterns, temperature, and salinity, enabling real-time forecasting of bluebottle presence with temporal continuity.

By incorporating environmental drivers and coastal characteristics using advanced 3D CNN architectures, we assess the relative importance of these factors. The project analyzes spatial patches offshore to capture broader spatiotemporal patterns of environmental change that influence bluebottle transport and beaching events.

---

## 🧰 Requirements

- **Python:** ≥ 3.10.18
- **TensorFlow:** ≥ 2.10.0 (GPU version recommended)
- **GPU:** NVIDIA GPU with 6-8GB VRAM (recommended for 3D CNN training)
- **Package Manager:** Anaconda/Miniconda
- **IDE:** VS Code or JupyterLab

> **Note:** This project uses GPU acceleration. CPU-only training is possible but significantly slower.

---

## 📁 Project File Structure

```
./
├── data/
│   ├── 2-Data/
│   │   └── Data files in original orginal format
│   └── processed/
│       ├── GoldCoast/
│       │   └── (processed data files)
│       └── Sydney_Newcastle/
│           └── (processed data files)
├── figures/                          # Plots and visualizations
├── notebooks/                        # Jupyter notebooks (EDA, analysis)
├── bluebottle.yaml                   # Conda environment file
└── README.md                         # This file
```

---

## 🗂️ Data Folder Setup

Create the required folder structure and place your data files:

### Windows (Command Prompt)
```cmd
mkdir data
```

Then place your data files in under `data/2-Data/`.

```cmd
mkdir data\processed
mkdir data\processed\GoldCoast
mkdir data\processed\Sydney_Newcastle
```




---

## ⚙️ Environment Setup

1. **Clone the repository:**
   ```bash
   git clone <repo_url>
   cd <repo_name>
   ```

2. **Create and activate the Conda environment:**
   ```bash
   conda env create -f bluebottle.yaml
   conda activate bluebottle
   ```

3. **Verify TensorFlow and GPU:**
   ```bash
   python -c "import tensorflow as tf; print('TF version:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))"
   ```


---

## 🧠 Model Details

| Model | Parameters | Key Feature |
|-------|-----------|-------------|
| **C3D** | ~200K | Uniform 3×3×3 kernels, fast training |
| **3D ResNet** | ~350K | Skip connections, deeper learning |
| **I3D** | ~300K | Multi-scale temporal convolutions |

**Input:** Temporal sequences of 7 consecutive days  
**Shape:** (7, 15, 15, 6) = (time_steps, height, width, channels)  
**Output:** Binary prediction (bluebottle stings: yes/no)

---


## 🔧 Quick Troubleshooting

**GPU not detected:**
```bash
nvidia-smi  # Check CUDA installation
```

**Out of Memory error:**  
Reduce `BATCH_SIZE` and `TEMPORAL_LENGTH` in the training script.

**Jupyter kernel not found:**
```bash
python -m ipykernel install --user --name=bluebottle
```

