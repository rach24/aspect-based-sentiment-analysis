# aspect-based-sentiment-analysis
The Task: Given an aspect term (also called opinion target) in a sentence, predict the sentiment label for the aspect term in the sentence.

The description of each column is as fellows: Column A: review sentence id Column B: review sentence Column C: aspect term in the sentence Column D: aspect term location Column E: sentiment label

Typical Workflow:

Pre-process the data files to normalize the data.
Build classifier model.
Evaluate the performance of the classifier model and report results.
Evaluation:

Results are generated via 10-fold cross validation.
Computed following metrics by taking average over 10 folds- Accuracy and avg. precision, avg. recall and avg. F1 scores for each of the three classes- { positive, negative, neutral } and for each of the two training dataset {data_1, data_2}.
