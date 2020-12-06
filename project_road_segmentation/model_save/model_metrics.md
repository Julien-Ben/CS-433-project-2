# Metrics

| Name | F1 | Accuracy | Precision | Recall | Model characteristic | Epoch | Loss | Optimizer | Dropout |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | 
| `baseline` | 0.835855   | 0.939475    | 0.764942 | 0.92126 | nothing since it's a baseline | 250 | `dice_coef_loss ` | `adam` | 0.15 |
| `dropout0_4` | 0.752763   | 0.900119    | 0.642347 | 0.909019 | dropout rate = 0.4 | 250 | `dice_coef_loss ` | `adam` | 0.4 |
| `dropout0` | 0.812501   | 0.933193    | 0.763074 | 0.871064 | dropout rate = 0 | 250 | `dice_coef_loss ` | `adam` | 0 |
| `dropout0_25` | 0.799207   | 0.922851    | 0.707707 | 0.917881 | dropout rate = 0.25 | 250 | `dice_coef_loss ` | `adam` | 0.25 |

0.799207	0.922851	0.707707	0.917881
