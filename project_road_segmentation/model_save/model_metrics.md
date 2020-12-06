# Metrics

| Name | F1 | Accuracy | Precision | Recall | Model characteristic | Epoch | Loss | Optimizer | Dropout | Number of filters |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | 
| `baseline` | 0.835855   | 0.939475    | 0.764942 | 0.92126 | nothing since it's a baseline | 250 | `dice_coef_loss ` | `adam` | 0.15 | 16 |
| `dropout0_4` | 0.752763   | 0.900119    | 0.642347 | 0.909019 | dropout rate = 0.4 | 250 | `dice_coef_loss ` | `adam` | 0.4 | 16 |
| `dropout0` | 0.812501   | 0.933193    | 0.763074 | 0.871064 | dropout rate = 0 | 250 | `dice_coef_loss ` | `adam` | 0 | 16 |
| `dropout0_25` | 0.799207   | 0.922851    | 0.707707 | 0.917881 | dropout rate = 0.25 | 250 | `dice_coef_loss ` | `adam` | 0.25 | 16 |
| `unet_fixed` | __0.860647__   | 0.949646    | 0.801232 | 0.92958 | fixed typo in `block_conv2d` AIcrowd submission of 0.894 f1| 250 | `dice_coef_loss ` | `adam` | 0.1 | 16 |
| `tversky_dice` | 0.858529   | 0.948796    | 0.798125 | 0.928825 | Used the tversky based `dice loss`. Same settings as `unet_fixed`| 250 | `dice_loss ` | `adam` | 0.1 | 16 |
| `tversky_alpha_0_4` | 0.857381   | 0.949133    | 0.80732 | 0.914062 | `focal_tversky` loss, $\alpha=0.4$, $\gamma = 0.75$| 250 | `focal_tversky_loss ` | `adam` | 0.1 | 16 |
| `tversky_alpha_0_45` | 0.855491   | 0.948298    | 0.803327 | 0.914899 | `focal_tversky` loss, $\alpha=0.45$, $\gamma = 0.75$| 250 | `focal_tversky_loss ` | `adam` | 0.1 | 16 |
| `tversky_alpha_0_6` | __0.855903__   | 0.947693    | 0.793696 | 0.92869 | `focal_tversky` loss, $\alpha=0.6$, $\gamma = 0.75$| 250 | `focal_tversky_loss ` | `adam` | 0.1 | 16 |
| `tversky_alpha_0_7` | 0.848423   | 0.944167    | 0.777108 | 0.934148 | `focal_tversky` loss, $\alpha=0.7$, $\gamma = 0.75$| 250 | `focal_tversky_loss ` | `adam` | 0.1 | 16 |
| `tversky_gamma_1` | 0.850091   | 0.945247    | 0.78418 | 0.928099 | `focal_tversky` loss, $\alpha=0.6$, $\gamma = 1$| 250 | `focal_tversky_loss ` | `adam` | 0.1 | 16 |
| `tversky_gamma_1_2` | 0.848839   | 0.944709    | 0.782063 | 0.928083 | `focal_tversky` loss, $\alpha=0.6$, $\gamma = 1.2$| 250 | `focal_tversky_loss ` | `adam` | 0.1 | 16 |

0.855213	0.947701	0.796415	0.923385
0.859936	0.949252	0.798714	0.931323 #dropout_0_15
0.84235	0.943748	0.792862	0.898428 #dropout 0


While `tversky_alpha_0_6` isn't best, it has graphical good results therefore we set $\alpha = 0.6$.