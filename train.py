import time
from typing import Optional
from models.seg_only_model import SegOnlyModel
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from models.base_model import BaseModel
from util.visualizer import Visualizer

import wandb
import wandb.sdk.wandb_run


def get_run_name(opt):
    components = [opt.name, opt.model]
    project_name = "-".join([str(c) for c in components])
    return project_name


def train(opt, seg_only_model: Optional[SegOnlyModel] = None) -> BaseModel:
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print("#training images = %d" % dataset_size)

    model = create_model(opt, seg_only_model)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(
                    model.get_current_visuals(), epoch, save_result
                )

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(
                        epoch, float(epoch_iter) / dataset_size, opt, losses
                    )

            if total_steps % opt.save_latest_freq == 0:
                print(
                    "saving the latest model (epoch %d, total_steps %d)"
                    % (epoch, total_steps)
                )
                save_suffix = "iter_%d" % total_steps if opt.save_by_iter else "latest"
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print(
                "saving the model at the end of epoch %d, iters %d"
                % (epoch, total_steps)
            )
            model.save_networks("latest")
            model.save_networks(epoch)

        print(
            "End of epoch %d / %d \t Time Taken: %d sec"
            % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time)
        )
        model.update_learning_rate()

    return model


if __name__ == "__main__":
    opt = TrainOptions().parse()

    run = wandb.init(project="instagan", name=get_run_name(opt), config=opt)

    name = opt.name
    model = opt.model

    if model != "insta_gan":
        raise Exception("Model not found")

    opt.name = name + "_seg"
    opt.model = "seg_only"
    opt.isTrain = False  # force the loading of the checkpoint
    seg_only_model = create_model(opt)
    seg_only_model.setup(opt)
    assert type(seg_only_model) is SegOnlyModel
    seg_only_model.eval()

    opt.name = name
    opt.model = "insta_gan"
    opt.isTrain = True
    final_model = train(opt, seg_only_model)

    assert type(run) is wandb.sdk.wandb_run.Run
    run.finish()
