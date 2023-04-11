from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

# Specify the path to your custom corpus
data_folder = "C:\\Users\\shayr_slxnwzf\\PycharmProjects\\pythonversion"

# Specify the column format of your corpus (in this example, 'text' and 'ner' columns)
columns = {0: 'text', 1: 'ner'}

# Load your corpus
corpus = ColumnCorpus(data_folder, columns, train_file='CONLL2_1.txt', dev_file='CONLL2_2.txt', test_file='CONLL2_3.txt')
'''
print(len(corpus.train))
print(corpus.train[0].to_tagged_string('ner'))
'''
# Create the word embeddings
embedding_types = [
    WordEmbeddings('glove'),
    WordEmbeddings('crawl')
]
embeddings = StackedEmbeddings(embedding_types)

# Create the tagger
tagger : SequenceTagger = SequenceTagger(hidden_size=256,
                                       embeddings=embeddings,
                                       tag_dictionary=corpus.make_label_dictionary(label_type='ner'),
                                       tag_type='ner',
                                       use_crf=True)
print(tagger)

# Train the tagger
trainer = ModelTrainer(tagger, corpus)
trainer.train('C:\\Users\\shayr_slxnwzf\\PycharmProjects\\pythonversion', learning_rate=0.1, mini_batch_size=32, max_epochs=150)
