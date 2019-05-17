import argparse

import torch
from tqdm import tqdm

import data_loader.data_loaders as module_data
import embedding.embedding as module_embedding
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from utils.util import generate_text


def main(config, resume):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        vocab_size=config['data_loader']['args']['vocab_size'],
        batch_size=512,
        seq_length=20,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2,
    )

    # build model architecture
    config["embedding"]["args"].update({"vocab": data_loader.dataset.vocab})
    embedding = config.initialize('embedding', module_embedding)

    config["arch"]["args"].update({"vocab": data_loader.dataset.vocab})
    config["arch"]["args"].update({"embedding": embedding})
    model = config.initialize('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(resume))
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    for temperature in [0.8, 1.0]:
        print('----- Temperatue: {} -----'.format(temperature))
        print(generate_text(
            model,
            start_seq='rick: I hate ',
            vocab=data_loader.dataset.vocab,
            temperature=temperature,
            length=15))

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            target = target.reshape(-1)
            if target.size()[0] != data_loader.batch_size:
                break

            data, target = data.to(device), target.to(device)
            output = model(data)

            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PyTorch Natural Language Processing Template')

    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')

    config = ConfigParser(parser)
    args = parser.parse_args()
    main(config, args.resume)
