import lightgbm as lgb

from pytorch_utils import *
from lightning_utils import *
from pytorch_vision_utils import *


models_path = Path('selected_models')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# The order used in the model; what would be returned by [os.path.basename(p) for p in Path(fr'{data_path}\test').glob('*')]
classes = ['ace of clubs', 'ace of diamonds', 'ace of hearts', 'ace of spades',
           'eight of clubs', 'eight of diamonds', 'eight of hearts', 'eight of spades',
           'five of clubs', 'five of diamonds', 'five of hearts', 'five of spades',
           'four of clubs', 'four of diamonds', 'four of hearts', 'four of spades',
           'jack of clubs', 'jack of diamonds', 'jack of hearts', 'jack of spades',
           'joker', 'king of clubs', 'king of diamonds', 'king of hearts', 'king of spades',
           'nine of clubs', 'nine of diamonds', 'nine of hearts', 'nine of spades',
           'queen of clubs', 'queen of diamonds', 'queen of hearts', 'queen of spades',
           'seven of clubs', 'seven of diamonds', 'seven of hearts', 'seven of spades',
           'six of clubs', 'six of diamonds', 'six of hearts', 'six of spades',
           'ten of clubs', 'ten of diamonds', 'ten of hearts', 'ten of spades',
           'three of clubs', 'three of diamonds', 'three of hearts', 'three of spades',
           'two of clubs', 'two of diamonds', 'two of hearts', 'two of spades']



### Classification-layer-retrained NN

# RexNet 1.5
class_retrain_model_name, class_retrain_extra = 'RexNet15', '0_First_Adam001_10_epochs'
class_retrain_model = timm.create_model('rexnet_150.nav_in1k', pretrained = True, num_classes = 53).eval().to(device)
class_retrain_transforms = timm.data.create_transform(**timm.data.resolve_model_data_config(class_retrain_model), is_training = False)
for param in class_retrain_model.features.parameters(): param.requires_grad = False
for param in class_retrain_model.stem.parameters(): param.requires_grad = False

# model.classifier
class_retrain_model.load_state_dict(torch.load(models_path / f'{class_retrain_model_name}_{class_retrain_extra}.pth', map_location = device))


### Fully-retrained NN


# RexNet 1.0
# full_retrain_model_name, full_retrain_extra = 'RexNet10', '0_First_Adam001_10_epochs'
full_retrain_experiment_name, full_retrain_model_name, full_retrain_extra = 'FullRetrain_EarlyStop', 'RexNet10', 'Adam001_max10_epochs'
# full_retrain_experiment_name, full_retrain_model_name, full_retrain_extra = 'ClassRetrain_EarlyStop', 'RexNet10', 'Adam001_max10_epochs'
full_retrain_model = timm.create_model('rexnet_100.nav_in1k', pretrained = True, num_classes = 53).eval().to(device)
full_retrain_transforms = timm.data.create_transform(**timm.data.resolve_model_data_config(full_retrain_model), is_training = False)
for param in full_retrain_model.features.parameters(): param.requires_grad = False
for param in full_retrain_model.stem.parameters(): param.requires_grad = False

# # RexNet 1.5
# full_retrain_model_name, full_retrain_extra = 'RexNet15', '0_First_Adam001_10_epochs'
# full_retrain_model = timm.create_model('rexnet_150.nav_in1k', pretrained = True, num_classes = 53).eval().to(device)
# transforms = timm.data.create_transform(**timm.data.resolve_model_data_config(full_retrain_model), is_training = False)
# for param in full_retrain_model.features.parameters(): param.requires_grad = False
# for param in full_retrain_model.stem.parameters(): param.requires_grad = False

full_retrain_model.load_state_dict(torch.load(models_path / f'{full_retrain_experiment_name}_{full_retrain_model_name}_{full_retrain_extra}.pth', map_location = device))


# Use the pred_image_class function from pytorch_vision_utils.py



### NN Feature Extraction -> Gradient Boosting

## Import the feature extraction model

# feats_model_name = 'RexNet10'
# feats_model_name = timm.create_model('rexnet_100.nav_in1k', pretrained = True, num_classes = 53).eval().to(device)

feats_model_name = 'RexNet15'
feats_model = timm.create_model('rexnet_150.nav_in1k', pretrained = True, num_classes = 53).eval().to(device)

feats_transforms = timm.data.create_transform(**timm.data.resolve_model_data_config(feats_model), is_training = False)

# No training and only feature extraction (up to pooling after final convolution, i.e. for RexNet 1.0 [batch, 1280, 7, 7] -> [batch, 1280], and 1920 for RexNet 1.5)
for param in feats_model.parameters(): param.requires_grad = False
feats_model = nn.Sequential(OrderedDict(stem = feats_model.stem, features = feats_model.features, pool = feats_model.head.global_pool))


## Import the Gradient Boosting model

num_iterations = 100
boosting_type = 'gbdt' # 'gbdt' vs 'dart' (dart also comes with more parameters: max_drop, skip_drop, xgboost_dart_mode, uniform_drop)
data_sample_strategy = 'bagging' # 'bagging' vs 'goss' (goss also comes with more parameters: top_rate, other_rate)

gb_model = lgb.Booster(model_file = models_path / f'{feats_model_name}_features_in_{num_iterations}_{data_sample_strategy}_{boosting_type}_lgbm.txt')


# Processing functions

def image_to_features(image: Image, model: torch.nn.Module, transform: tv.transforms.Compose,
                      device: torch.device = 'cuda' if torch.cuda.is_available() else 'cpu') -> torch.Tensor:
    model.eval()
    with torch.inference_mode(): feats = model(transform(image).unsqueeze(0).to(device)).squeeze().to('cpu')
    return feats

def gb_predict_classes(feats: torch.Tensor, model: lgb.Booster, class_names: list[str]) -> dict[str, float]:
    '''Return the (ordered) predicted probabilities of each class for the given image
    '''
    probs = model.predict([feats], num_iteration = gb_model.best_iteration) # Already probabilities, not logits
    return OrderedDict(sorted({class_names[i]: float(probs[0][i]) for i in range(len(class_names))}.items(), key = itemgetter(1), reverse = True))
    # class_id = torch.argmax(probs, dim = 1)
    # return class_names[class_id.cpu()], probs.unsqueeze(0).max().cpu().item()


