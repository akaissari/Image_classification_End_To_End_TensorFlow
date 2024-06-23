from ImageClassifier.config.configuration import ConfigurationManager
from ImageClassifier.components.prepare_callbacks import Callback
from ImageClassifier.components.evaluation import Evaluation
from ImageClassifier import logger



STAGE_NAME = "Evaluation"


class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        val_config = config.get_validation_config()
        evaluator = Evaluation(config=val_config)
        evaluator.evaluation()
        evaluator.save_score()




if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
        