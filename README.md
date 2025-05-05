# Multi-Modal-Few-Shot-Learning-for-Anthesis-Prediction-of-Individual-Wheat-Plants
This is share code for 'Multi-Modal Few-Shot Learning for Anthesis Prediction of Individual Wheat Plants'

Anthesis prediction is crucial for breeding wheat. While current tools provide estimates of average anthesis at the field scale, they fail to address the needs of breeders who require accurate predictions for individual plants. Hybrid breeders have to finalize their plans for pollination at least 10 days before such flowering is due and biotechnology field trials in the United States and Australia must report to regulators 7 to 14 days before the first plant flowers. Currently, predicting anthesis of individual wheat plants is a labour-intensive, inefficient, and costly process. Individual wheat of the same cultivar within the same field may exhibit substantial variations in anthesis timing, due to significant variations in their immediate surroundings. In this study, we proposed a machine vision solution to predict anthesis of individual wheat plants that is accurate, efficient and economical. We use a multimodal approach that combines imagery with in-situ meteorological measurements to accurately forecast anthesis in individual plants in the field. We developed a novel approach that simplifies the anthesis prediction problem into binary or three-class classification tasks, which aligns better with breeders’ requirements in individual wheat flowering prediction on the crucial days before anthesis. Furthermore, we incorporated a few-shot learning method to improve the model's adaptability across different growth environments and to address the challenge of limited training data. The model achieved an F1 score above 0.8 in all planting settings.

Workflow structure

![image](https://github.com/user-attachments/assets/265d7e12-0dc3-43c6-be6b-ae5381eb1aa5)

![image](https://github.com/user-attachments/assets/7cf0766c-e5df-4c60-b429-efc4d9b28796)

![image](https://github.com/user-attachments/assets/844606a1-a063-4a2f-af70-00779ede7b64)

Result figure
![image](https://github.com/user-attachments/assets/b768ba9e-fd31-406a-8b61-b8619874140e)

![image](https://github.com/user-attachments/assets/b29b456e-ef6a-4fba-a637-615e2e55c8d9)

IWC_dataset_generate_two_class_1.py: used to generate the IWC dataset for meta-learning.

metal_learning_two_classes_2.py: contains the meta-learning implementation.

evaluate_metal_learning_model_3.py: used to evaluate the trained meta-learning model.

few_shot_learning_inference_4.py: performs inference using few-shot learning.

Few-shot learning one-step TRF and Few-shot learning one-step TRF (no TRF anchor) are used to test different anchor configurations in few-shot learning.

Few-shot learning (no weather) is an ablation study. Do not use models that include weather input for this test.

Hints:

Dataset TPA 1 2023 corresponds to the Early dataset.

Dataset TPA 2 2023 corresponds to the Mid dataset.

Dataset BC 2023 corresponds to the Late dataset.

Dataset TRF 2023 corresponds to the Rosedale dataset.

A regular expression pattern is used in the code:
pattern = re.compile(fr'^{days}_([0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}})_(?:BC|TPA|TRF)-ID{plant_id}_IMG_\d+\.jp[e]?g$', re.IGNORECASE)
If you wish to train and test on your own custom dataset, please modify the regular expression accordingly to match your file naming format.
