from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators.problem import Text2TextProblem
from tensor2tensor.utils import registry
import pandas as pd
@registry.register_problem
class TranslationQuery(Text2TextProblem):
    """Predict next line of poetry from the last line. From Gutenberg texts."""

    @property
    def approx_vocab_size(self):
        return 2**30  # ~8k

    @property
    def is_generate_per_split(self):
    # generate_data will shard the data into TRAIN and EVAL for us.
        return False

    @property
    def dataset_splits(self):
        """Splits of data to produce and number of output shards for each."""
        # 10% evaluation data
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 9,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir
        del dataset_split

        dataset_file = "yahoo.csv"
        reader = pd.read_csv(dataset_file,header=0,delimiter=",",chunksize=100000)
        for df in reader:
            for row in df.itertuples():
                input = row[1]
                target = row[2]
                yield {
                    "inputs": input,
                    "targets": target,
                }
