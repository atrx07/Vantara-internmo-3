# PRD_TRACEABILITY.md — Vantara PRD-to-Build Matrix

This matrix maps the source PRD to the locked implementation sequence/evidence. It is a check against omission, not a replacement for the source PRD.

| PRD area | Locked implementation | Primary step | Evidence |
|---|---|---:|---|
| Executive outputs | churn, CLV, next purchase, recommendations, segments | 04–08 | API/dashboard/model evidence |
| 3.1 reproducible pipeline | `python -m src.pipeline` | 01–03 | pipeline run |
| 3.1 churn models | six classical + ANN | 04–06 | MLflow/comparison |
| 3.1 sequence DL | LSTM 30-day purchase | 05 | metrics/artifacts |
| 3.1 anomaly DL | autoencoder | 05 | reconstruction evidence |
| 3.1 clustering | K-Means + GMM | 04 | metrics/profiles |
| 3.1 explainability | SHAP/LIME/PDP/plain text | 06/08 | XAI assets/dashboard |
| 3.1 REST/dashboard | FastAPI + Streamlit | 07–08 | tests/UI |
| 3.2 churn AUC >=.80 | final held-out test | 06 | final metrics |
| 3.2 CLV R² >=.60 | final regression evaluation | 06 | final metrics |
| 3.2 churn recall >=.70 | threshold/model selection | 06 | final metrics |
| 3.2 API p95 <400 ms | benchmark | 09 | benchmark report |
| 3.2 dashboard <3s | benchmark | 09 | benchmark report |
| 3.2 scripted pipeline | no notebook-only production logic | 03/09 | clean run |
| 4.1 Logistic Regression | sklearn | 04 | run log |
| 4.1 Decision Tree | sklearn | 04 | run log |
| 4.1 Random Forest | sklearn | 04 | run log |
| 4.1 XGBoost | xgboost | 04 | run log |
| 4.1 LightGBM | lightgbm | 04 | run log |
| 4.1 KNN/SVM | locked SVM | 04 | run log |
| 4.1 ANN | PyTorch | 05 | run log |
| 4.1 LSTM | PyTorch | 05 | run log |
| 4.1 Autoencoder | PyTorch | 05 | run log |
| 4.1 K-Means + second | K-Means + GMM | 04 | cluster report |
| 4.1 SHAP + LIME | full production churn XAI | 06 | plots/report |
| 4.1 FastAPI/Streamlit | API + API-consuming UI | 07–08 | smoke/tests |
| 4.1 Docker | compose | 09 | clean compose run |
| 4.2 deferrals | remain out of scope | all | dependency/scope audit |
| 5 Online Retail II | exact supplied workbook | 00–01 | hash/manifest |
| 5 missing IDs | product-only retain/customer exclude | 02 | tests/stats |
| 5 returns | explicit flags/features | 02 | tests |
| 5 price issues | flag/document rule | 02 | tests/audit |
| 5 admin StockCodes | exclude product behavior | 02 | tests/config |
| 5 seasonality | feature + EDA | 02–03 | feature/notebook |
| 6.1 both sheets | loader | 01 | loader test |
| 6.2 duplicate/outlier/description | cleaning pipeline | 02 | tests/audit |
| 6.2 validation | Pandera + business checks | 01–02 | failing/passing tests |
| 6.3 Pandas/NumPy reusable | `src/` pipeline | 01–03 | code/pipeline |
| 6.3 low-cardinality OHE | preprocessing | 02/04 | artifact/config |
| 6.3 StockCode target/frequency | frequency-derived product popularity | 02 | feature tests |
| 6.3 scaling | scale-sensitive only | 04–05 | pipelines |
| 6.3 persisted transforms | artifacts | 04–06 | reload tests |
| 6.4 EDA | required analyses/hypotheses | 03 | `01_eda.ipynb` |
| 6.4 VIF | correlation/VIF gate | 03 | report/notebook |
| 7 all required features | `FEATURE_CONTRACT.md` | 02 | feature table/tests |
| 7 leakage | snapshot strictness + attack tests | 02 onward | leakage suite |
| 8.1 tuning | Randomized/CV, early stopping | 04 | logs |
| 8.2 ANN requirements | BN/dropout/early stop | 05 | architecture/loss |
| 8.2 LSTM event sequence | amount/gap/category | 05 | sequence tests/model |
| 8.2 autoencoder | scaled spending/recon error | 05 | artifact/report |
| 8.3 seed/split | 42, 70/15/15 | 02 | split artifact |
| 8.3 train-only CV | locked | 04–06 | code/logs |
| 8.3 test once | separate final evaluation | 06 | status/final artifact |
| 8.3 run logging | MLflow local | 04–06 | exported run table |
| 9 K selection | elbow + silhouette | 04 | cluster report |
| 9 second clustering | GMM assumption test | 04 | report |
| 9 readable labels | profile names/stats | 04/08 | report/UI |
| 10 SHAP global/local | required representatives | 06 | XAI assets |
| 10 LIME compare | same customer | 06 | report |
| 10 PDP | top 2–3 | 06 | plots |
| 10 plain-language | deterministic template | 06/08 | API/UI |
| 11 churn metrics | all six + ANN | 04–06 | comparison |
| 11 CLV metrics | MAE/RMSE/R² | 04/06 | comparison |
| 11 clustering metrics | silhouette/DB | 04 | report |
| 11 5-fold supervised CV | classification/regression/grouped adaptation | 04–05 | logs |
| 11 consolidated table | generated | 06 | report |
| 12.1 required API | 4 required endpoints | 07 | API tests |
| 12.1 PostgreSQL | predictions/segments | 07 | integration tests |
| 12.1 Pydantic | request schemas | 07 | invalid tests |
| 12.2 segmentation filters | Streamlit | 08 | UI test |
| 12.2 churn leaderboard | risk × value | 08 | UI |
| 12.2 revenue forecast | Holt-Winters | 08 | UI/report |
| 12.2 customer SHAP | explorer | 08 | UI |
| 12.2 batch + PDF/CSV | API/UI | 07–08 | flow test |
| 12.3 one-command compose | DB/API/UI | 09 | clean run |
| 13 type hints/docstrings | code standard | all | lint/review |
| 13 structured logs | logging | 01 onward | code/tests |
| 13 >=70 coverage | Pytest coverage | 09 | report |
| 13 YAML config | config | 01 onward | source audit |
| 13 reproducibility | pinned env/seed | 01/09 | clean run |
| 13 secrets | env/.gitignore | 01/09 | audit |
| 14 architecture/ER | docs | 09 | PNGs |
| 15 repository structure | preserve PRD paths | 01 onward | tree |
| 16 math appendix | actual equations/curves | 09 | final report |
| 17 week plan | compressed equivalent roadmap | 00–09 | roadmap |
| 18 deliverables | complete set | 09 | inventory |
| 19 risks | imbalance/leakage/timeline | all | implementation/report |
| 20 acceptance criteria | final matrix | 09 | acceptance report |
| 21 tools | locked approved stack | 01 | requirements |
