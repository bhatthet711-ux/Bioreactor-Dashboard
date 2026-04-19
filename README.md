# 🧫 BioReactor Analytics Suite

An interactive, multi-axis dashboard for visualizing bioprocess engineering data over a 48-hour fermentation cycle. Designed for process monitoring, phase detection, and specific growth rate (μ) analysis.

## Setup Instructions

1. **Install Dependencies:**
   Ensure you have Python 3.9+ installed. Run the following command in your terminal:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application:**
   Execute the Streamlit application using this command in your project directory:
   ```bash
   streamlit run app.py
   ```

3. **Provide Data:**
   - Upload your own `CSV` matching the required schema (`time_hr`, `pH`, `temp_C`, `DO_percent`, `OD600`).
   - Or, simply click **"Use Demo Dataset"** in the sidebar to simulate a realistic run.

## Features

- **Multi-Axis Interactive Chart:** Combines up to 5 scaled metrics (OD600, pH, DO, Temperature, and Growth Rate μ) into a unified Plotly graph, preventing visual clutter while enabling scale-agnostic comparisons.
- **Automated Phase Detection:** Evaluates biomass (OD600) profiles to dynamically overlay Lag, Log, and Stationary growth phases using shaded backgrounds.
- **Statistical Summaries & Anomalies:** Generates a real-time table populated with metrics standard derivations, highlighting instances where process bounds exceeded 2-sigma thresholds.
- **Process Warning Banners:** Proactively surfaces warnings if Critical Process Parameters (CPPs) such as pH, DO (<20%), and Temp (>1°C deviation) violate biological thresholds.
- **In-depth Correlation Analysis:** Displays a Pearson heatmap providing insights into metabolic coupling (e.g. tracking DO drops synchronized to exponential growth).

## Engineering Context: Fermentation & Monod Kinetics

In large-scale biomanufacturing, strict monitoring of Critical Process Parameters (CPPs) is essential to maximizing cellular yield and product titer. This dashboard mathematically assesses growth rates through Monod kinetics, specifically calculating specific growth rate (`μ`) as a time-derivative of the natural logarithm of biomass (`d(ln(X))/dt`). During a batch fermentation cycle, microorganisms navigate distinct physiological boundaries: the **Lag** phase (adaptation), the **Log** phase (exponential division causing rapid Dissolved Oxygen depletion and lowering pH via acid byproducts), and the **Stationary** phase (nutrient exhaustion). By tracking these variables concurrently, engineers can predict limits and optimize feed strategies before process stresses occur.
