# Metrics

| Name | F1 | Accuracy | Precision | Recall | Model characteristic | Epoch | Loss | Optimizer | Dropout | Number of filters |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | 
| `baseline` | 0.835855   | 0.939475    | 0.764942 | 0.92126 | nothing since it's a baseline | 250 | `dice_coef_loss ` | `adam` | 0.15 | 16 |
| `dropout0_4` | 0.752763   | 0.900119    | 0.642347 | 0.909019 | dropout rate = 0.4 | 250 | `dice_coef_loss ` | `adam` | 0.4 | 16 |
| `dropout0` | 0.812501   | 0.933193    | 0.763074 | 0.871064 | dropout rate = 0 | 250 | `dice_coef_loss ` | `adam` | 0 | 16 |
| `dropout0_25` | 0.799207   | 0.922851    | 0.707707 | 0.917881 | dropout rate = 0.25 | 250 | `dice_coef_loss ` | `adam` | 0.25 | 16 |
| `unet_fixed` | 0.860647   | 0.949646    | 0.801232 | 0.92958 | fixed typo in `block_conv2d` AIcrowd submission of 0.894 f1| 250 | `dice_coef_loss ` | `adam` | 0.1 | 16 |
| `tversky_dice` | 0.858529   | 0.948796    | 0.798125 | 0.928825 | Used the tversky based dice loss. Same settings as `unet_fixed`| 250 | `dice_loss ` | `adam` | 0.1 | 16 |
| `tversky_alpha_0_4` | 0.857381   | 0.949133    | 0.80732 | 0.914062 | `focal_tversky`, $\alpha=0.4$, $\gamma = 0.75$| 250 | `focal_tversky_loss ` | `adam` | 0.1 | 16 |
| `tversky_alpha_0_45` | 0.806248   | 0.929914    | 0.749888 | 0.871768 | `focal_tversky`, $\alpha=0.45$, $\gamma = 0.75$| 250 | `focal_tversky_loss ` | `adam` | 0.1 | 16 |
