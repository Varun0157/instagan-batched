import copy

import os
import time
from typing import Optional
from models.seg_only_model import SegOnlyModel
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from models.base_model import BaseModel
from util import html
from util.visualizer import Visualizer
from util.visualizer import save_images

import wandb
import wandb.sdk.wandb_run


def get_run_name(opt):
    components = [opt.name, opt.model]
    project_name = "-".join([str(c) for c in components])
    return project_name


def train(opt, seg_only_model: SegOnlyModel) -> BaseModel:
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


def test_seg_only(model: SegOnlyModel, opt) -> None:
    opt.results_dir = "./results/"
    opt.phase = "test"
    opt.aspect_ratio = 1.0
    opt.name = opt.name + "_seg"

    opt.num_threads = 1  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, "%s_%s" % (opt.phase, opt.epoch))
    webpage = html.HTML(
        web_dir,
        "Experiment = %s, Phase = %s, Epoch = %s" % (opt.name, opt.phase, opt.epoch),
    )

    for i, data in enumerate(dataset):
        if i >= 50:
            break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        if i % 5 == 0:
            print("processing (%04d)-th image... %s" % (i, img_path))
        save_images(
            webpage,
            visuals,
            img_path,
            aspect_ratio=opt.aspect_ratio,
            width=opt.display_winsize,
        )
    # save the website
    webpage.save()


if __name__ == "__main__":
    opt = TrainOptions().parse()

    run = wandb.init(project="instagan", name=get_run_name(opt), config=opt)

    name = opt.name
    model = opt.model

    opt.name = name + "_seg"
    opt.model = "seg_only"
    opt.continue_train = True
    seg_only_model = create_model(opt)
    seg_only_model.setup(opt)
    assert type(seg_only_model) is SegOnlyModel
    for param in seg_only_model.netG_A.parameters():
        param.requires_grad = False
    for param in seg_only_model.netG_B.parameters():
        param.requires_grad = False
    seg_only_model.eval()

    for param in seg_only_model.netG_A.parameters():
        print(param.data)
        break
    for param in seg_only_model.netG_B.parameters():
        print(param.data)
        break

    # opt_copy = copy.deepcopy(opt)
    # test_seg_only(seg_only_model, opt_copy)

    opt.name = name
    opt.model = "insta_gan"
    opt.continue_train = False
    # final_model = train(opt, seg_only_model)

    for param in seg_only_model.netG_A.parameters():
        print(param.data)
        break
    for param in seg_only_model.netG_B.parameters():
        print(param.data)
        break

    opt_copy = copy.deepcopy(opt)
    test_seg_only(seg_only_model, opt_copy)

    assert type(run) is wandb.sdk.wandb_run.Run
    run.finish()
