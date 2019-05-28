import torch
import torch.nn as nn
import torchvision.models as models


class ResNet101:
    def __init__(
        self,
        feature_extract=True,
        num_classes=0,
        use_pretrained=False
    ):
        self._feature_extract = feature_extract
        self._use_pretrained = use_pretrained
        self._num_classes = num_classes
        self._model = self._initialize_model()

    def _initialize_model(self):
        msg = 'pretarin' if self._use_pretrained else 'unpretrain'
        print("Start to create {} model.....".format(msg))
        resnet101 = models.resnet101(pretrained=self._use_pretrained)

        if self._feature_extract:
            self._set_parameter_requires_grad(resnet101)

        fc_in_feature = resnet101.fc.in_features
        resnet101.fc = nn.Linear(fc_in_feature, self._num_classes)

        return resnet101

    def _set_parameter_requires_grad(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def get_model(self):
        return self._model
