import datasets
import pandas as pd

class MOH(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIG_CLASS=datasets.BuilderConfig

    def _info(self):

        feature = {
            'id':datasets.Value('int32'),
            'sent_id':datasets.Value('string'),
            'word_index': datasets.Value('int32'),
            'tokens':datasets.Sequence(datasets.Value('string')),
            'label':datasets.ClassLabel(num_classes=2, names=['Not Metaphor', 'Is Metaphor'])
        }

        return datasets.DatasetInfo(
            description='Vua metaphor detection datasets.',
            features=datasets.Features(feature),
            config_name=self.config.name
        )
    
    def _split_generators(self, dl_manager):

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={'filepath': self.config.data_files}),
        ]
    
    def _generate_examples(self, filepath):
        df = pd.read_csv(filepath, sep='\t')

        for index, row in df.iterrows():
            if row.isna().any():
                continue
            tokens = row['sentence'].split()
            word_idx = 0
            for index, token in tokens:
                if token.startswith('<b>') and token.endswith('</b>'):
                    token.strip('<b>').strip('</b>')
                    word_idx=index
            yield index, {
                'id': index,
                'sent_id': index,
                'tokens': tokens,
                'word_index': word_idx,
                'label': 0 if row['class'] == 'literal' else 1
            }