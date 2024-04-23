import gradio as gr
import os
from pathlib import Path
from PIL import Image

from ingredients import (classes, pred_image_classes,
                         class_retrain_model, class_retrain_transforms,
                         full_retrain_model, full_retrain_transforms,
                         feats_model, feats_transforms,
                         gb_model, image_to_features, gb_predict_classes)


def multiple_predictions(image: Image) -> dict[str, dict[str, float]]:
    full_retrain_preds = pred_image_classes(image, full_retrain_model, full_retrain_transforms, classes)

    feats = image_to_features(image, feats_model, feats_transforms)
    feat_extr_to_gb_preds = gb_predict_classes(feats, gb_model, classes)

    class_retrain_preds = pred_image_classes(image, class_retrain_model, class_retrain_transforms, classes)

    return full_retrain_preds, feat_extr_to_gb_preds, class_retrain_preds



# Create the Gradio demo
gr.Interface(fn = multiple_predictions,
    inputs = gr.Image(type = 'pil'),
    outputs = [gr.Label(num_top_classes = 3, label = 'Fully-retrained RexNet-1.0 predictions'),
               gr.Label(num_top_classes = 3, label = 'RexNet-1.5 feature extraction -> LightGMB classifier predictions'),
               gr.Label(num_top_classes = 3, label = 'Classification-layer-retrained RexNet-1.5 predictions')],
    examples = [[Path('examples') / example] for example in os.listdir('examples')],
    cache_examples = False, # This would avoid invoking the chatbot for the example queries (it would invokes it on them on startup instead)
    title = 'Card Image Classifier Comparison',
    description = '''A comparison of card image classifiers, showing the training and performance benefits of gradient boosting over NN classification layers:
    * A fully-retrained RexNet-1.0  --  ~20min to train  --  F1 score of ~1 on unseen test data (it is overkill after all)
    * A compound model of RexNet-1.5 features extraction followed by a LightGBM classifier  --  ~5min to train  --  F1 score of 0.5433 on unseen test data 
    * A classification-layer-retrained RexNet-1.5  --  ~9min to train  --  F1 score of 0.4884 on unseen test data'''
).launch()


