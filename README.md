# mlfs-book
O'Reilly book - Building Machine Learning Systems with a feature store: batch, real-time, and LLMs


## ML System 


[Dashboards for Our ML Systems](https://kkkk25343.github.io/mlfs-book/)

    
# Create a conda or virtual environment for your project before you install the requirements
    pip install -r requirements.txt
    pip install -r requirements-cc.txt


##  Run pipelines with make commands
    Use "cat Makefile" to see make commands with their corresponding ipynb notebool

    make aq-features:
        ipython notebooks/airquality/1_air_quality_feature_backfill.ipynb
    make aq-train
        ipython notebooks/airquality/3_air_quality_training_pipeline.ipynb
    make aq-inference
        ipython notebooks/airquality/2_air_quality_feature_pipeline.ipynb
        ipython notebooks/airquality/4_air_quality_batch_inference.ipynb
    make aq-clean: 
         python mlfs/clean_hopsworks_resources.py aq
         This is to clean Feature Groups. When we want to add one more feature which is the mean pm2.5 for the previous 3 days, we have to clean features first. If not, when training XBoost, the input is ambiguous and causes ERROR.
         
    aq-llm:
        ipython notebooks/airquality/5_function_calling.ipynb 
        We tried it but LLM is too large to download. But it runs.

or 
    make aq-all


## City we choose 
We choose Xiaotong's hometown, Guilin. The longyinluxiaoxue station was where she used to go to primary school. Data were fetched from 2014 to 2025. The settings of AQICN are:

    AQICN_URL=https://api.waqi.info/feed/@7181
    AQICN_COUNTRY=china
    AQICN_CITY=guilin
    AQICN_STREET=longyinluxiaoxue

## Feature added (for Grade C)
We add a new feature new feature `pm25_rolling_mean_3d` which represents the mean PM2.5 value over the previous 3 days. But some days may not have PM2.5 measurements. Here are our rules. We only use days that have valid PM2.5 values. For each day, we find the previous 3 days that have valid PM2.5 measurements. If there are fewer than 3 valid previous days, we use all available valid days. Days without PM2.5 values will have NaN for this feature and will be excluded from training.

We also editted `/mlfs/airquality/util.py` to adjust new feature when we trained the model, or the variables were not compatible. 

## 3 Ways to run the code
1. **Using Make commands**  
   Run the entire pipeline (backfill → features → train → inference).
2. **Directly running Jupyter Notebook**
3. **GitHub Action**
   
   Click `Settings` in the Github menu and setup API keys for HOPSWORKS and AQICN.
   <img width="1576" height="372" alt="image" src="https://github.com/user-attachments/assets/b64cf311-5719-40ba-931e-fd9cc18d74db" />
   Click `Actions` in the Github menu and run `air-quality-daily` workflow. Remember to check `air-quality-daily.yml` for configuration first. 



## Feldera


mkdir -p /tmp/c.app.hopsworks.ai
ln -s  /tmp/c.app.hopsworks.ai ~/hopsworks
docker run -p 8080:8080 \
  -v ~/hopsworks:/tmp/c.app.hopsworks.ai \
  --tty --rm -it ghcr.io/feldera/pipeline-manager:latest


