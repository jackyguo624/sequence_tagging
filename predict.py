import tensorflow as tf
from model.config import Config
from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
import os

def main():
    config = Config()
    model = NERModel(config)
    model.build()
    model.restore_session('./results/test/model.weights/')

    if not os.path.exists(config.predict_output):
        os.makedirs(config.predict_output)
    ofile = open(config.predict_output+'/result', 'w')
    test = CoNLLDataset(config.filename_test, None)
    for words, _ in test:
        preds = model.predict(words)
        for line in zip(words, preds):
            ofile.write("{0} {1}\n".format(line[0], line[1]))
        ofile.write("\n")
    ofile.close()


if __name__ == "__main__":
    main()