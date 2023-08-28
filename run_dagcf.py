from logging import getLogger
from recbole.utils import init_logger, init_seed
from recbole.trainer import Trainer
from recbole.trainer import DATrainer
from recbole.config import Config
from recbole.model import BPR
from recbole.data import create_dataset, data_preparation, prepare_DA_data_matrix


if __name__ == '__main__':
     
    config = Config(model='BPR', dataset='ml-100k')
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # prepare DA_data_matrix
    if config['DA_matrix_path'] is not None:
        train_data_matrix = np.load(config['DA_matrix_path'])
    else:
        train_data_matrix = prepare_DA_data_matrix(train_data)

    # model loading and initialization
    model = BPR(config, train_data.dataset, train_data_matrix).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = DATrainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    # model evaluation
    test_result = trainer.evaluate(test_data)

    logger.info(f'best valid result: {best_valid_result}')
    logger.info(f'test result: {test_result}')
