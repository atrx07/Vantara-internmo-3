# SHAP–LIME borderline-customer comparison

The shared customer has validation risk `0.1954750490` against the frozen threshold `0.1954750490`. SHAP and LIME share `8` of their top `10` resolved features: category_affinity_02, customer_tenure_days, frequency_orders, net_spend, recency_days, seasonal_purchase_concentration, unique_product_count, variance_interpurchase_gap_days. Differences are expected because SHAP attributes the fitted forest prediction while LIME fits a local surrogate around one customer. Both outputs are descriptive and do not imply causality.
