#     Preeclampsia-Research-Project
## Demo
https://huggingface.co/spaces/doeygirl93/preeclampsia_screener

## Overveiw
Preeclampsia is a serious pregnacy disorder and disportionetly impacts women of color. If not detected early. Preeclampsia can lead to severe outcomes for both the mother and baby.  This project explores an AI-driven system can help identify high‑risk cases for early PE screening by utilizing retinal fundus images 
This project was orginally trained in kaggle hence why i uploaded it. I used pytorch to train the model and gradio for the demo


## Awards won
I've quallified to the International Hosa research conference for my research and got 3rd place in washington
## Abstract
Preeclampsia (PE) is a leading cause of maternal morbidity, with Black women in the United States facing a 60% higher risk and more severe outcomes. Traditional screening methods often fail to identify over 40% of positive cases and rely on subjective, race-based risk factors. This research proposed a failure-aware, pigment-invariant AI system for early PE screening by utilizing retinal fundus images to detect hypertensive retinopathy biomarkers. A DenseNet-201 CNN was utilized with a two-phase transfer learning approach on over 36,000 images. To address racial disparities, a pigment-invariant pipeline was engineered using standardized color normalization and rigorous data augmentations to ignore retinal pigmentation. For clinical safety, real time quality gates using OpenCV Laplacian Variance and resolution monitoring were implemented to reject low quality diagnostic images. On an independent test set, the model achieved 98.0% sensitivity and a 0.96 AUC. The system was deployed as a web application delivering results in under 500ms, featuring Grad-CAM heatmaps for visual evidence and clinical accountability. These results demonstrate a scalable, low-cost, and equitable solution to the maternal health crisis that significantly outperforms traditional screening detection rates.


https://drive.google.com/file/d/1gZbOciZcPVWQRucdkBLzHkTcA7luKAg6/view?usp=drive_link
