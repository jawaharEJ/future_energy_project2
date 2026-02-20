# Semester 5 Feature: Advanced Data Visualization & Custom Reports

## 🎨 New Features Implemented

### 1. **Interactive 3D Charts** (`/visualization`)
- **3D Scatter Plot**: Visualize Year, Month, and Energy Consumption in 3D space
- Hover for detailed information
- Rotate, zoom, and pan with mouse controls
- Color-coded by energy consumption values

### 2. **Heatmap Analysis** (`/visualization`)
- **Year vs Month Heatmap**: Identify seasonal patterns
- Shows average energy consumption for each month across all years
- Color intensity represents consumption levels
- Great for spotting peak/off-peak seasons

### 3. **Bubble Chart Visualization** (`/visualization`)
- **Population vs Energy**: Relationship between population and consumption
- Bubble size represents industrial growth percentage
- Color gradient shows year progression
- Reveals correlation between growth and energy usage

### 4. **Distribution Box Plot** (`/visualization`)
- **Energy by Month**: Statistical distribution analysis
- Shows median, quartiles, and outliers
- Identify which months have most variable consumption
- Useful for understanding consumption patterns

### 5. **Custom Report Builder** (`/reports`)
- **Advanced Filtering**:
  - Filter by year range (start/end year)
  - Filter by specific month
  - Combine multiple filters

- **Comprehensive Metrics**:
  - Total records in filtered dataset
  - Average, max, and min consumption
  - Standard deviation (variability)
  - Revenue projections in Indian Rupees (₹)
  - Average population and industrial growth

- **Statistical Summary Table**:
  - Count, mean, std, min, quartiles, max
  - Professional data analysis format
  - Easy to understand distribution

- **Export Functionality**:
  - Download filtered reports as CSV
  - Ready for Excel/further analysis
  - Timestamped filenames

## 📊 Technical Implementation

### Backend Routes Added:
```python
GET  /visualization           - Main visualization page
GET  /api/3d-chart           - 3D scatter plot (Plotly)
GET  /api/heatmap            - Heatmap visualization
GET  /api/population-consumption-bubble - Bubble chart
GET  /api/box-plot           - Distribution box plot
GET  /reports                - Custom report builder
POST /api/generate-report    - Generate filtered report
POST /api/export-report      - Export report as CSV
```

### New Dependencies:
- **plotly**: Interactive web-based visualizations
- **seaborn**: Statistical data visualization
- **numpy**: Numerical computations

## 🚀 How to Use

### Access Visualizations:
1. Navigate to **Visualizations** in the main menu
2. Click through tabs to explore:
   - 3D Scatter (default)
   - Heatmap (seasonal patterns)
   - Bubble Chart (relationships)
   - Distribution (statistics)

### Generate Custom Reports:
1. Go to **Reports** in the main menu
2. Set filters:
   - Enter start year (e.g., 2020)
   - Enter end year (e.g., 2024)
   - Optionally select a specific month
3. Click **Generate Report**
4. View statistics and metrics
5. Click **Export CSV** to download

## 💡 Use Cases

| Feature | Use Case |
|---------|----------|
| 3D Chart | Overall trend visualization across multiple dimensions |
| Heatmap | Identify seasonal peaks/troughs and planning |
| Bubble Chart | Analyze growth impact on energy consumption |
| Box Plot | Understand consumption variability by month |
| Reports | Detailed analysis for specific periods/months |

## 🔄 Integration with Existing Features

- All visualizations work with existing energy data
- Revenue calculations in **Indian Rupees (₹)** @ ₹10/kWh
- Compatible with current prediction system
- Works with user authentication system
- Maintains session security

## 📈 Future Enhancements

- Real-time data updates
- Comparative benchmarking
- Anomaly detection alerts
- Machine learning forecasting
- IoT sensor integration

---
**Created**: January 2026 | **For**: Semester 5 Project
