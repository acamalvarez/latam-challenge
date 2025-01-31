# Description

This repository contains a solution for the latam machine engineer challenge. Some of the work done on the files were:

The file exploration.ipynb was refactored by using black formatter and by updating the figures so that they are more descriptive.

Xgboost was added to requirements.txt.

Test model was modified so the predict method works correctly.

The modules are compliant with the black formatter.

Updated dependencies to latest versions to avoid installation problems in docker

# API

The API is hosted in https://latam-challenge-616721223966.us-central1.run.app. To check for the status go to `/health` and for the documentation go to `/docs`.

The API uses the model logistic regression with feature importance and balance. It showed a recall of 0.52 and a precision of 0.88 for class 0 and recall of 0.25 and 0.69 for class 1, which compared to the metrics of other models, shows a better balance in terms of handling both classes. It has similar metrics than the model XGBoost with Feature Importance and Balance, however, the logistic regression one was chosen because usually if more efficient as it requires less computational resources and it is generally easy to interpret.

# How to run
To run the project locally you can:
## First option:
- Clone the repository
- Create a virtual environment
- Checkout to virtual environment and install dependencies by using pip: `pip install -r requirements.txt -r requirements-dev.txt -r requirements-test.txt`
- To run the api run `uvicorn challenge.api:app`. Use the –reload flag if you want changes in the code to restart the server and reflect the changes in the application.

## Second option
- Use docker `docker
