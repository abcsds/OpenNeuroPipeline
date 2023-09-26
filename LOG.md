# 25 Sep
Setup pipeline to run with "ds004324".
Running to evaluate results.

# 26 Sep
Pipeline running. SVM hangs. Optuna doesn't timeout with SVMs somehow.

Possible upgrades:
check for hung processes (multiprosses, or celery)
restart hung processes (celery?)
logging (celery?)
Result and model storage (mlflow?)