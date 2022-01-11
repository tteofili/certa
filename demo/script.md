DEMO SCRIPT
===========

# Explaining ER predictions with CERTA 
0. Explanation visualization
   1. visualize how a given open triangle issues a flip by altering different attributes
1. Train DeepER, DeepMatcher and Ditto on the following dataset kinds:
   1. Dataset with few attributes
   2. Dataset with many attributes
   3. Dirty dataset
   4. Long text dataset
2. **DIFFERENT RATIONALES**<br/> for each dataset: 
   1. take _n_ pairs from the test set that are correctly classified as _matching_ by all models
   2. take _n_ pairs from the test set that are correctly classified as _non-matching_ by all models
   3. generate explanations with _CERTA_, _DiCE_ and _Mojito_ for all predictions
   4. show how different models pay attention to different attributes
      1. show different CERTA saliencies for each model and same prediction 
      2. show Mojito saliencies 
      3. compare faithfulness and confidence indication
      4. alter attributes (based on human intuition) identified by explanation and see how the model score is affected
      5. show different CERTA counterfactuals for each model and same prediction
      6. show DiCE counterfactuals for each model and same prediction
      7. compare sparsity, diversity, proximity
3. **COUNTERINTUITIVE PREDICTIONS**<br/> for each dataset:
   1. take _n_ pairs from the test set that are wrongly classified as _matching_ by at least one model
   2. take _n_ pairs from the test set that are wrongly classified as _non-matching_ by at least one models
   3. generate explanations with _CERTA_, _DiCE_ and _Mojito_ for all predictions
   4. show how different models pay attention to different attributes
      1. show different saliencies highlighting differences between explanations for correct and wrong predictions
      2. compare faithfulness and confidence indication for saliencies
      3. alter attributes (based on human intuition) identified by explanation and see how the model score is affected
      4. show CERTA counterfactuals for wrong predictions
      5. show DiCE counterfactuals for wrong predictions
      6. compare sparsity, diversity, proximity
4. **ALTERED TRAINING SETS**<br/> for each dataset:
   1. train two models (e.g. DeepMatcher and Ditto) with:
      1. standard training set
      2. an "important" attribute (according to human intuition) filtered out
      3. two attributes merged
      4. **ER-properties-augmented** training set
   2. explain predictions using differently trained models for the following input pair kinds (from the test set):
      1. different prediction for the same input pair (e.g. filtered-out-trained-model would more likely get things wrong according to human intuition)
      2. almost identity input pair
      3. identity input pair
      4. same predictions according to merged-attributes-trained-model and standard model 
         1. e.g. for Ditto it makes sense, from a neural network architecture perspective
         2. e.g. for DeepMatcher they should more likely differ as there's an attribute embedding piece in the NN architecture
