import argparse

import torch
from tqdm import tqdm

import data_loader.data_loaders as module_data
import embedding.embedding as module_embedding
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import utils.util as module_util
from parse_config import ConfigParser


def main(config, resume):
    logger = config.get_logger("test")

    # setup data_loader instances
    data_loader = config.initialize(
        "test_data_loader", module_data, **{"training": False}
    )

    # build model architecture
    try:
        config["embedding"]["args"].update({"vocab": data_loader.dataset.vocab})
        embedding = config.initialize("embedding", module_embedding)
    except:
        embedding = None
    config["arch"]["args"].update({"vocab": data_loader.dataset.vocab})
    config["arch"]["args"].update({"embedding": embedding})
    model = config.initialize("arch", module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config["loss"])
    metric_fns = [getattr(module_metric, met) for met in config["metrics"]]
    test_fns = [getattr(module_util, method) for method in config["test_methods"]]

    logger.info("Loading checkpoint: {} ...".format(resume))
    checkpoint = torch.load(resume)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    total_loss = {t_fn.__name__: 0.0 for t_fn in test_fns}
    total_metrics = {t_fn.__name__: torch.zeros(len(metric_fns)) for t_fn in test_fns}

    with torch.no_grad():
        for batch_idx, (input_ids, attention_mask, segment_ids, target) in enumerate(
            tqdm(data_loader)
        ):
            target = target.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            segment_ids = segment_ids.to(device)

            for test_method in test_fns:
                output = test_method(model, (input_ids, attention_mask, segment_ids))

                # computing loss, metrics on test set
                loss = loss_fn(output, target)
                batch_size = input_ids.shape[0]
                total_loss[test_method.__name__] += loss.item() * batch_size
                for i, metric in enumerate(metric_fns):
                    total_metrics[test_method.__name__][i] += (
                        metric(output, target) * batch_size
                    )

            output = model(batch=(input_ids, attention_mask, segment_ids))

    n_samples = len(data_loader.sampler)
    for test_method in test_fns:
        log = {
            f"{test_method.__name__}_loss": total_loss[test_method.__name__] / n_samples
        }
        log.update(
            {
                f"{test_method.__name__}_{met.__name__}": total_metrics[
                    test_method.__name__
                ][i].item()
                / n_samples
                for i, met in enumerate(metric_fns)
            }
        )
        logger.info(log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PyTorch Natural Language Processing Template"
    )

    parser.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    parser.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    config = ConfigParser(parser)
    args = parser.parse_args()
    main(config, args.resume)
