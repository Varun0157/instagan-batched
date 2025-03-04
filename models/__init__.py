import importlib
from typing import Optional
from models.base_model import BaseModel
from models.insta_gan_model import InstaGANModel
from models.seg_only_model import SegOnlyModel


def find_model_using_name(model_name):
    # Given the option --model [modelname],
    # the file "models/modelname_model.py"
    # will be imported.
    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)

    # In the file, the class called ModelNameModel() will
    # be instantiated. It has to be a subclass of BaseModel,
    # and it is case-insensitive.
    model = None
    target_model_name = model_name.replace("_", "") + "model"
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print(
            "In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase."
            % (model_filename, target_model_name)
        )
        exit(0)

    return model


def get_option_setter(model_name):
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options


def create_model(opt, seg_model: Optional[SegOnlyModel] = None):
    model = find_model_using_name(opt.model)
    instance = model()
    if type(instance) is InstaGANModel:
        print("creating InstaGANModel")
        print("type(seg_model):", type(seg_model))
        assert seg_model is not None
        instance.initialize(opt, seg_model)
    else:
        instance.initialize(opt)
    print("model [%s] was created" % (instance.name()))
    return instance
