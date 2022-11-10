|Model|CV|LB| EPOCHS | Transforms |
 |---|---|---|---|---|
|efficientnet-b0|0.92|0.209433|8|rm_vt_flip|
|efficientnet-b6|0.95|0.212622|8|rm_vt_flip|
|efficientnet-b6|Bad|Bad|8|rm_hr_flip|
|efficientnet-b6| 0.95 | 0.212623 |8|rm_vt_flip, rm_shap|
|efficientnet-b6| Bad|Bad |8|RandomVerticalFlip, RandomAdjustSharpness, TrivialAugmentWide|
|efficientnet-b6| Bad|Bad |8|RandomVerticalFlip, TrivialAugmentWide|
|efficientnet-b6| Bad|Bad |8|RandomVerticalFlip, Normalization|
|efficientnet-b6| --|-- |8|RandomVerticalFlip, Normalization, 
| RESTART | RESTART | RESTART | RESTART | RESTART |
|3_efficientnet-b6| 0.95 | 0.99038 | 8 |rm_vt_flip, rm_shap|
|3_efficientnet-b6| -- | -- | 8 | rm_vt_flip, rm_shap, Colorjitter|
|efficientnet-b6 (sgd)| 0.98 | -- | 50 | rm_vt_flip(0.5) / rm_hr_flip(0.5) / rm_sharp(2) |
|efficientnet-b6 (adamw)| 0.99 | -- | 10 | rm_vt_flip(0.5) / rm_hr_flip(0.5) / rm_sharp(2) |
|efficientnet-b6 (adamw)| 0.99 | -- | 10 | rm_vt_flip(0.5) / rm_hr_flip(0.5) / rm_sharp(2) / label smoothing |
|efficientnet-v2_m (adamw)| 0.99 | -- | 10 | rm_vt_flip(0.5) / rm_hr_flip(0.5) / rm_sharp(2) |
|efficientnet-v2_m (adamw)| 0.99 | -- | 10 | rm_vt_flip(0.5) / rm_hr_flip(0.5) / rm_sharp(2) / label smoothing(0.1) |

