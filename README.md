# COLIEE 2021 

This repository contains the code to reproduce NeuralMind's submissions to COLIEE 2021 presented in papers [Yes, BM25 is a Strong Baseline for Legal Case Retrieval](https://arxiv.org/abs/2105.05686) (Task 1) and [To Tune or Not To Tune? Zero-shot Models for Legal Case Entailment](https://arxiv.org/abs/2202.03120) (Task 2). 

COLIEE is a legal case competition that evaluates automated systems on legal tasks.


# Task 1: Legal Case Retrieval

The task of Legal Case Retrieval involves reading a new case Q, and extracting supporting cases S1, S2, ... Sn for the decision of Q from the entire case law corpus. The dataset for task 1 is composed of predominantly Federal Court of Canada case laws, and it is provided as a pool of cases containing 4415 documents. The input is an unseen legal case, and the output is the relevant cases extracted from the pool that support the decision of the input case.


## Results

|               Model                    |      F1       | 
| ---------------------------------------| ------------- | 
| Median of submissions in competition   |     2.79      |
| 3rd best submission of competition     |     4.56      |
| BM25  (ours)                           |     9.37      |
| Best submission of competition         |    19.17      |

Our vanilla BM25 is a good baseline for the task as it achieves second place in the competition and its F1 score is well above the median of submissions. This result is
not a surprise since it agrees with results from other competitions, such as the Health Misinformation and Precision Medicine tracks of TREC 2020 [13]. The advantage of our approach is the simplicity of our method, requiring only the document’s segmentation and the grid search. One of the disadvantages is the time spent during the retrieval of segmented documents


# Task 2: Legal Case Entailment

The task of Legal Case Entailment involves the identification of a paragraph from existing cases that entails the decision of a new case.

Given a decision Q of a new case and a relevant case R, a specific paragraph that entails the decision Q needs to be identified. This task requires one to identify a paragraph which entails the decision of Q, so a specific entailment method is required which compares the meaning of each paragraph in R and Q in this task.


## To Tune or Not To Tune? Zero-shot Models for Legal Case Entailment

In this work, we investigate this transfer ability to the legal domain. For that, we participated in the legal case entailment task of COLIEE 2021, in which we use such models with no adaptations to the target domain. Our submissions achieved the highest scores, surpassing the second-best submission by more than six percentage points. Our experiments confirm a counter-intuitive result in the new paradigm of pretrained language models: that given limited labeled data, models with little or no adaption to the target task can be more robust to changes in the data distribution and perform better on held-out datasets than models fine-tuned on it.

## Models

**monoT5-zero-shot**: We use a model T5 Large fine-tuned on MS MARCO, a dataset of approximately 530k query and relevant passage pairs. We use a checkpoint available at Huggingface’smodel hub that was trained with a learning rate of 10−3 using batches of 128 examples for 10k steps, or approximately one epoch of the MS MARCO dataset. In each batch, a roughly equal number of positive and negative examples are sampled.

**monoT5**: We further fine-tune monoT5-zero-shot on the COLIEE 2020 training set following a similar training procedure described for monoT5-zero-shot. The model is fine-tuned with a learning rate of 10−3 for 80 steps using batches of size 128, which corresponds to 20 epochs. Each batch has the same number of positive and negative examples.

**DeBERTa**: Decoding-enhanced BERT with disentangled attention(DeBERTa) improves on the original BERT and RoBERTa architectures by introducing two techniques: the disentangled attention mechanism and an enhanced mask decoder. Both improvements seek to introduce positional information to the pretraining procedure, both in terms of the absolute position of a token and the relative position between them. We fine-tune DeBERTa on the COLIEE 2020 training set following a similar training procedure described for monoT5. 

**DebertaT5 (Ensemble)**: We use the following method to combine the predictions of monoT5 and DeBERTa (both fine-tuned on COLIEE 2020 dataset): We concatenate the final set of paragraphs selected by each model and remove duplicates, preserving the highest score. It is important to note that our method does not combine scores between models. The final answer for each test example is composed of individual answers from one or both models. It ensures that only answers with a certain degree of confidence are maintained, which generally leads to an increase in precision.


## Results

| Model                           |  Train data   |   Evaluation    |     F1       | Description
| ------------------------------- | ------------- | --------------- | ------------ | ------------ | 
| Median of submissions           |               |     Coliee      |    58.60     |              |
| Coliee 2nd best team            |               |     Coliee      |    62.74     |              |
| DeBERTa (ours)                  |    Coliee     |     Coliee      |    63.39     | Single model |
| monoT5  (ours)                  |    Coliee     |     Coliee      |    66.10     | Single model |
| monoT5-zero-shot (ours)         |   MS Marco    |     Coliee      |    68.72     | Single model |
| DebertaT5 (ours)                |    Coliee     |     Coliee      |    69.12     |   Ensemble   |

In this table, we present the results. Our main finding is that our zero-shot model achieved the best result of a single model on 2021 test data, outperforming DeBERTa and monoT5, which were fine-tuned on the COLIEE dataset. As far as we know, this is the first time that a zero-shot model outperforms fine-tuned models in the task of legal case entailment. Given limited annotated data for fine-tuning and a held-out test data, such as the COLIEE dataset, our results suggest that a zero-shot model fine-tuned on a large out-of-domain dataset may be more robust to changes in data distribution and may generalize better on unseen data than models fine-tuned on a small domain-specific dataset. Moreover, our ensemble method effectively combines DeBERTa and monoT5 predictions,achieving the best score among all submissions (row 6). It is important to note that despite the performance of DebertaT5 being the best in the COLIEE competition, the ensemble method requires training time, computational resources and perhaps also data augmentation to perform well on the task, while monoT5-zero-shot does not need any adaptation. The model is available online and ready to use.


## Conclusion

Based on those results, we question the common assumption that it is necessary to have labeled training data on the target domain to perform well on a task. Our results suggest that fine-tuning on a large labeled dataset may be enough.

# COLIEE 2022 

This repository contains the code to reproduce NeuralMind's submissions to COLIEE 2022 presented in the paper [Billions of Parameters Are Worth More Than In-domain Training Data: A case study in the Legal Case Entailment Task](https://arxiv.org/abs/2105.05686) (Task 2)

## Billions of Parameters Are Worth More Than In-domain Training Data: A case study in the Legal Case Entailment Task

Recent work has shown that language models scaled to billions of parameters, such as GPT-3, perform remarkably well in zero-shot and few-shot scenarios. In this work, we experiment with zero-shot models in the legal case entailment task of the COLIEE 2022 competition. 
Our experiments show that scaling the number of parameters in a language model improves the F1 score of our previous zero-shot result by more than 6 points, suggesting that stronger zero-shot capability may be a characteristic of larger models, at least for this task. Our 3B-parameter zero-shot model outperforms all models, including ensembles, in the COLIEE 2021 test set and also achieves the best performance of a single model in the COLIEE 2022 competition, second only to the ensemble composed of the 3B model itself and a smaller version of the same model. Despite the challenges posed by large language models, mainly due to latency constraints in real-time applications, we provide a demonstration of our zero-shot monoT5-3b model being used in production as a search engine, including for legal documents. The demo of our system are available at [neuralsearchx.neuralmind.ai](https://neuralsearchx.neuralmind.ai).

## Results

| Model                           |  Train data   |   Evaluation    |     F1       | Description
| ------------------------------- | ------------- | --------------- | ------------ | ------------ | 
| Median of submissions           |               |     Coliee      |    63.91     |              |
| monoT5-base-zero-shot (ours)    |   MS Marco    |     Coliee      |    63.25     | Single model |
| Coliee 2nd best team            |               |     Coliee      |    66.94     |              |
| monoT5-3B-zero-shot  (ours)     |   MS Marco    |     Coliee      |    67.57     | Single model |
| monoT5-Ensemble-zero-shot (ours)|   MS Marco    |     Coliee      |    67.83     |   Ensemble   |

For the COLIEE 2022 competition, we submitted three runs using different sizes of monoT5-zero-shot models: a monoT5-base (row 2), a monoT5-3B (row 4) and an ensemble between monoT5-base and monoT5-3B (row 5). Performance consistently increases with model size and two of our submissions (rows 4 and 5) score above the median of submissions and above all teams in the competition. 
Furthermore, our ensemble method effectively combines the predictions of different monoT5 models, achieving the best performance among all submissions (row 5).

## Conclusion

In this work, we explored the zero-shot ability of a multi-billion parameter language model in the legal domain. We showed that, for the legal case entailment task, language models without any fine-tuning on the target dataset and target domain can outperform models fine-tuned on the task itself. Furthermore, our results support the hypothesis that scaling language models to billions of parameters improves zero-shot performance. This method has the potential to be extended to other legal tasks, such as legal information retrieval and legal question answering, especially in limited annotated data scenarios.


## How do I get the dataset?

Those who wish to use previous COLIEE data for a trial, please contact rabelo(at)ualberta.ca.


## How do I evaluate?

As our best model is a zero-shot one, we provide only the evaluation script.
- [Task 1 notebook](https://colab.research.google.com/drive/1jFew3w-aGu0mp8iVB78IjpTVm3W5w-Pl?usp=sharing) (Test set 2021)
- [Task 2 notebook](https://colab.research.google.com/drive/1bB4YiJm7_de0bsPJOWv9V0HYGj8_IzwU?usp=sharing) (Test set 2021)

To reproduce our COLIEE 2022 results, use the Task 2 notebook above but with the [monoT5-3B model](https://huggingface.co/castorini/monot5-3b-msmarco-10k) and the COLIEE 2022 dataset. At least 25GB of RAM and a Tesla P100 GPU are required.

## References

[1] [Document Ranking with a Pretrained Sequence-to-Sequence Model](https://arxiv.org/abs/2003.06713)

[2] [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654)

[3] [ICAIL '21: Proceedings of the Eighteenth International Conference on Artificial Intelligence and Law](https://dl.acm.org/doi/10.1145/3462757.3466103)

[4] [Proceedings of the Eighth International Competition on Legal Information Extraction/Entailment](https://sites.ualberta.ca/~rabelo/COLIEE2021/COLIEE2021proceedings.pdf)


## How do I cite this work?

~~~ {.xml
 @article{bm25_baseline,
    title={Yes, BM25 is a Strong Baseline for Legal Case Retrieval},
    author={Moraes, Guilherme and Rodrigues, Ruan and Lotufo, Roberto and Nogueira, Rodrigo},
    journal={Proceedings of the Eighth International Competition on Legal Information Extraction/Entailment},
    url={https://sites.ualberta.ca/~rabelo/COLIEE2021/COLIEE2021proceedings.pdf},
    year={2021}
}
~~~

~~~ {.xml
 @article{to_tune,
    title={To Tune or Not To Tune? Zero-shot Models for Legal Case Entailment},
    author={Moraes, Guilherme and Rodrigues, Ruan and Lotufo, Roberto and Nogueira, Rodrigo},
    journal={ICAIL '21: Proceedings of the Eighteenth International Conference on Artificial Intelligence and Law June 2021 Pages 295–300},
    url={https://dl.acm.org/doi/10.1145/3462757.3466103},
    year={2021}
}
~~~

~~~ {.xml
 @article{coliee_2022_NM,
    title={Billions of Parameters Are Worth More Than In-domain Training Data: A case study in the Legal Case Entailment Task},
    author={Moraes, Guilherme and Bonifacio, Luiz and Jeronymo, Vitor and Lotufo, Roberto and Nogueira, Rodrigo},
    journal={Proceedings of the Sixteenth International Workshop on Juris-informatics (JURISIN 2022)},
    url={},
    year={2022}
}
~~~

## Contact
If you have any questions, please email guilhermemr04@gmail.com
