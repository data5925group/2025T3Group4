# 🪼 Bluebottle Sting and Environmental Data Analysis

## 📘 Overview
This project investigates the relationship between **Bluebottle (Physalia physalis) sting events** and key **environmental variables** such as sea surface temperature (SST), salinity, ocean currents, and wind across coastal waters between **2010–2024**.

By integrating sting occurrence data with gridded environmental reanalysis datasets, the analysis identifies:
- Environmental **drivers** of sting events  
- **Temporal patterns** and **seasonal trends**  
- A predictive **Bluebottle Risk Index** to quantify sting likelihood  

---

## 📊 Data Description

### 🧪 Sting Dataset
| Property | Description |
|-----------|--------------|
| Shape | (797, 3) |
| Columns | `['time', 'stings_sum', 'stings_Binary']` |
| Date range | 1 Jan 2011 – 9 Dec 2023 |
| Mean daily stings | 32.9 |
| Binary distribution | 0 (no stings) = 570, 1 (stings present) = 227 |

---

### 🌊 Environmental Dataset
| Property | Description |
|-----------|--------------|
| Shape | (5479, 12) |
| Variables | `SST_MEAN`, `SALINITY_MEAN`, `CURRENT_SPEED_MEAN`, `WIND_SPEED_MEAN`, `U_WIND_MEAN`, `V_WIND_MEAN`, and standard deviations |
| Spatial coverage | 153.4°E–154.87°E, -27.75°N–-29.20°N |
| Temporal coverage | 1 Jan 2010 – 31 Dec 2024 |

---

### 🔗 Merged Dataset
| Property | Description |
|-----------|--------------|
| Shape | (744, 15) |
| Overlap period | 2010–2024 |
| Missing values | 0 |

---

## 📈 Key Results

### 🔹 Correlations (Pearson)
| Variable | r | Significance |
|-----------|---|--------------|
| SST Mean | 0.165 | *** |
| U-Wind Mean | -0.163 | *** |
| SST Variability | -0.132 | *** |
| Current Speed Mean | 0.103 | ** |
| Wind Speed Mean | 0.087 | * |

**Lag Effects:**  
SST shows the strongest positive correlation with sting events up to a **3-day lag**, indicating short-term environmental influence.

---

## 🕒 Temporal Patterns

### 📅 Monthly
- **Peak Activity:** January–February (SST > 26 °C, currents ≈ 0.55–0.60 m/s)  
- **Low Activity:** May–August  

### 🌦️ Seasonal Summary
| Season | Sting Probability | Mean SST (°C) |
|---------|------------------|----------------|
| Summer | 47% | 25.9 |
| Autumn | 16% | 25.8 |
| Spring | 9% | 23.0 |

---

## 🌪️ Extreme Event Analysis

| Threshold | Sting Count |
|------------|--------------|
| 90th percentile | 244 |
| 99th percentile | 831 |

**Environmental Conditions During Extremes:**
- SST ≈ +1.16 °C  
- Stronger westerly winds (**U-Wind ≈ –4.8 m/s**)  
- Reduced SST variability  

---

## ⚠️ Bluebottle Risk Scoring Framework

An **8-level Bluebottle Risk Index** was developed to classify sting risk:

| Risk Level | Score Range | Sting Probability |
|-------------|-------------|-------------------|
| Low | 0–2 | 9.5% |
| High | 6–8 | 56% |

**Predictive contrast:**  
High-risk days are **5.9× more likely** to experience sting events compared to low-risk days.

---

## 🧩 Summary

- **Warmer SSTs**, **moderate currents**, and **onshore winds** are key predictors of sting risk.  
- **Seasonal SST peaks** align with sting outbreaks.  
- The **Risk Index framework** effectively distinguishes **high- vs low-risk** periods using environmental data.

---

## 👩‍💻 Author
**Siddharth Choudhury**  
Master of Data Science and Decisions