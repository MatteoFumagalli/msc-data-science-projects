## Data Management Project
Environmental and Seasonal Effects on Football Match Performance:  
Analysis of the Big Five European Leagues over the Last Three Seasons

---

### :bar_chart: Project Description
This project, developed as part of the **Data Management** course within the MSc in Data Science, focuses on the full data management workflow for football matches. The pipeline includes acquisition, enrichment, storage, analysis, and quality.

---

### :dart: Objectives
The goal of this project is to investigate the relationship between weather conditions, seasonal effects, and soccer match performance. Thus, the main research questions are:
- Does weather influence match outcomes and match statistics?
- Do seasonal effects influence match outcomes and match statistics?

---

### :file_folder: Repository Structure
- `data/` :arrow_right: raw and preprocessed CSV files, as well as the final database file
- `src/` :arrow_right: Python modules for acquisition, enrichment, storage, analysis, and quality
- `.env.example` :arrow_right: template file with database credentials and API keys
- `environment.yml` :arrow_right: Python environment specification
- `notebook.ipynb` :arrow_right: main notebook orchestrating the pipeline
- `presentation.pdf` :arrow_right: exam presentation slides
- `report.pdf` :arrow_right: exam report

---

### :gear: Setup Instructions
1) Create Python environment
```bash
conda env create -f environment.yml
conda activate data_man
```

2) Configure .env variables
    - Copy `.env.example` to `.env`
    - Fill in PostgreSQL credentials and API keys

3) Load country polygons for complex spatial queries (this step must be executed after creating the PostgreSQL database and before running the data quality code)
    - Download polygons from [Natural Earth – 10m cultural vectors](https://www.naturalearthdata.com/downloads/10m-cultural-vectors/)
    - Extract the `.zip` folder and place the files in the `utils/` folder
    - From command prompt, execute the following instruction replacing all values inside `< >` with your local paths and credentials
      ```bash
      <path_to_shp2pgsql.exe> -I -s 4326 <path_to/utils/ne_10m_admin_0_countries.shp> countries_admin0 | <path_to_psql.exe> -U <user_postgres> -d <football_db>
      ```

---

### :arrow_forward: Project Execution
Open `notebook.ipynb` and run the cells using the play button