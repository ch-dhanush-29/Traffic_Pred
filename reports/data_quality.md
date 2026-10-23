# FlowCast: Data Quality & Cleaning Audit Report
**Northline Corridor Intelligent Transportation Systems Programme**  
*Document Version:* 1.0  
*Owner:* Data & AI Division, Meridian Mobility Systems  
*Dataset Period:* January 1, 2025 – May 31, 2025 (151 calendar days)  

---

## 1. Executive Summary
This report documents the rigorous data validation, quarantine, cleaning, and harmonization pipeline implemented for FlowCast v1.0. The corridor consists of **25 arterial road segments** monitored by inductive loop and radar detectors at **30-minute aggregation**. 

The raw data ingested from three operational feeds contained 7 documented real-world defects:
1. Exact sensor retry duplicates.
2. Physically impossible negative volume values.
3. Detector failure overspeed outliers (> 200 km/h).
4. Saturated/faulty detector over-occupancy (> 100%).
5. Sensor dropout missing windows and null telemetry.
6. Blank congestion levels (roughly 1 in 7 records).
7. Discordant date formats (DD/MM/YYYY vs YYYY-MM-DD) and non-standard categorical weather strings.

All defects were successfully quarantined and remediated in strict compliance with **PRD Section 9** and the **FlowCast Data Dictionary**.

---

## 2. Source Telemetry Ingestion Audit

| Table Name | Raw Grain | Initial Records | Valid Records | Quarantined / Dropped | Primary Remedy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`traffic_sensor_log`** | Segment × 30-min window | 178,468 | 176,701 | 1,767 duplicates | Deduplication on `[road_id, date, time]`; first entry retained |
| **`weather_observations`** | Station × Clock hour | 10,872 | 10,872 | 0 | Format harmonization and linear station interpolation |
| **`calendar_events`** | Whole corridor × Day | 151 | 151 | 0 | Direct calendar join on ISO date |

---

## 3. Detailed Data Defects & Remediation Protocol

### Defect 1: Exact and Key-Level Duplicates
- **Observation:** `1,767` records in `traffic_sensor_log.csv` were duplicate transmissions caused by detector polling retries.
- **Action:** Executed deduplication on primary composite key `(road_id, date, time)`. Retained the earliest verified telemetry packet.
- **Result:** Exact 0 duplicates remain; row count stabilized to 176,701.

### Defect 2: Impossible Negative Volume Outliers
- **Observation:** `241` records showed negative traffic volumes (e.g. -1 to -50) caused by inverted sensor loops.
- **Action:** Flagged and replaced negative volumes with `NaN`, then applied local linear time interpolation bounded by temporal segment medians.
- **Result:** Minimum traffic volume is 0.

### Defect 3: Extreme Speed Outliers
- **Observation:** `237` records reported speeds exceeding 200 km/h (some > 400 km/h) due to sensor timing glitches.
- **Action:** Speed capped at physical arterial maximum (120 km/h thresholding rule, strict quarantine above 200 km/h). Imputed via segment-by-window median speed.
- **Result:** Speeds strictly bounded in `[0, 120] km/h`, reflecting genuine urban arterial conditions.

### Defect 4: Saturated Occupancy Outliers
- **Observation:** `234` records had occupancy readings exceeding 100.0%.
- **Action:** Sensor saturation replaced with segment-window median occupancy.
- **Result:** Occupancy bounded strictly within `[0.0, 100.0]%`.

### Defect 5: Sensor Dropout and Missing Telemetry
- **Observation:** 4,387 missing volumes, 4,382 missing speeds, and 4,383 missing occupancy records.
- **Action:** Implemented a three-tier imputation protocol:
  1. *Tier 1 (Short gaps <= 2h):* Causal linear interpolation within segment.
  2. *Tier 2 (Medium gaps):* Segment $\times$ Time-of-Day median matching.
  3. *Tier 3 (Corridor baseline):* Segment general median fallback.
- **Result:** 0 null values in all predictive feature fields across the entire analysis-ready dataset.

### Defect 6: Missing Congestion Levels
- **Observation:** `26,624` records (~15% of the dataset) had blank `congestion_level` entries.
- **Action:** Derived congestion level using the official engineering capacity formula:
  $$\text{V/C} = \frac{\text{traffic\_volume}}{\text{road\_capacity} / 2}$$
  - $\text{V/C} < 0.50 \implies \textbf{Free-flow}$
  - $0.50 \le \text{V/C} < 0.80 \implies \textbf{Moderate}$
  - $0.80 \le \text{V/C} < 1.00 \implies \textbf{Heavy}$
  - $\text{V/C} \ge 1.00 \implies \textbf{Severe}$
- **Result:** 100% complete labels across all 176,701 records. Class distribution: Free-flow (59.6%), Moderate (25.1%), Heavy (9.8%), Severe (5.5%).

### Defect 7: Date Discrepancies and Inconsistent Categoricals
- **Observation:** Weather observations used `DD/MM/YYYY` while traffic used `YYYY-MM-DD`. Weather conditions included casing and whitespace variants (`'rain '`, `'RAIN'`, `'rainy'`, `'foggy'`).
- **Action:** Normalized all dates to ISO 8601 timestamps. Harmonized weather vocabulary to 4 canonical categories: `['Clear', 'Cloudy', 'Rain', 'Fog']`. Broadcasted hourly weather across both 30-minute windows in each clock hour.

---

## 4. Verification and Leakage Guard
- **Temporal Alignment:** Verified strict time-ordering per road segment before computing lag features.
- **No Lookahead Leakage:** All rolling statistics were generated with `shift(1)`, guaranteeing that telemetry at time $t$ uses strictly observations from $t-1, t-2, \dots$.
- **Validation Sign-Off:** The final analysis-ready dataset contains **176,701 records**, **44 predictive features**, and **zero missing values** in modeling matrices.
