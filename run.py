import yaml
import argparse
import logging
import os
from datetime import datetime
from module import Preprocessor, Trainer, Predictor

log_folder = './logs/'
filename = '{:%m-%d-%Y}'.format(datetime.now())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process configuration.')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--log_level', type=str, default="INFO")
    args = parser.parse_args()

    formatter = logging.Formatter('%(asctime)-15s %(message)s')
    logging.basicConfig(level=args.log_level)
    logger = logging.getLogger('global_logger')

    # create logs folder if not exist
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)

    fileHandler = logging.FileHandler(log_folder + filename, 'w', 'utf-8')
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    with open(args.config, 'r') as config_file:
        try:
            print('Preprocessing...')
            config = yaml.safe_load(config_file)
            preprocessor = Preprocessor(config['preprocessing'], logger)
            _, _, train_x, train_y, validate_x, validate_y, test_x = preprocessor.process()

            if config['training']['model_name'] != 'naivebayse':
                config['training']['vocab_size'] = len(preprocessor.word2ind.keys())

            print('Training...')
            pretrained_embedding = None
            trainer = Trainer(config['training'], logger, preprocessor.classes, pretrained_embedding)
            model, accuracy, cls_report, history = trainer.fit_and_validate(train_x, train_y, validate_x, validate_y)
            logger.info("accuracy:{}".format(accuracy))
            logger.info("\n{}\n".format(cls_report))

            print('Predicting...')
            predictor = Predictor(config['predict'], logger, model)
            probs = predictor.predict_prob(test_x)
            predictor.save_result(preprocessor.test_ids, probs)
        except yaml.YAMLError as err:
            logger.warning('Config file error: {}'.format(err))
