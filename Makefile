abs_data.csv: Project2_Interval_regression_parameters.R abs_features.csv abs_targets.csv linear_features.csv linear_targets.csv sin_features.csv sin_targets.csv
        R CMD BATCH Project2_Interval_regression_parameters.R
linear_data.csv: Project2_Interval_regression_parameters.R abs_features.csv abs_targets.csv linear_features.csv linear_targets.csv sin_features.csv sin_targets.csv
        R CMD BATCH Project2_Interval_regression_parameters.R
sin_data.csv: Project2_Interval_regression_parameters.R abs_features.csv abs_targets.csv linear_features.csv linear_targets.csv sin_features.csv sin_targets.csv
        R CMD BATCH Project2_Interval_regression_parameters.R
result.png: Week1.py abs_features.csv abs_data.csv abs_targets.csv linear_features.csv linear_data.csv linear_targets.csv sin_features.csv sin_data.csv sin_targets.csv
        python Week1.py
