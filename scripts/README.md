## How to Evaluate Uncertainty Predictions

1. Run `predict.py` to generate disparity predictions as well as depth uncertainty predictions (and optionally visualizations of disparity prediction errors and predicted disparity uncertainty).

1. Run `generate_depth_from_disparity.py` to generate depth maps from disparity predictions.

1. Run `evaluate_uncertainty_predictions.py` to evaluate the quality of the depth uncertainty predictions.

     * NOTE: Before running the above script:
        * You should have run the IVOA data preprocessing script on the test set to generate the image patch dataset that will be used to evaluate the uncertainty predictions.

        * If you want to use a calibration model on top of the originally predicted uncertainty estimates, you should have run the `calibrate_uncertainty_estimates.py` script on the training data to generate the calibration model.