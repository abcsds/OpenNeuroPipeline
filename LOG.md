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

# 27 Sep
Added report file: Removes subjects where k<=0, normalizes feature importance by k, and visualizes experiment-wide feature importance.

Posible upgrade:
Feature distance between subjects with k<=0 and k>0

Update pipeline: store features.