# COLIEE 2021 - task 2: Legal Case Entailment

This repository contains the code to reproduce NeuralMind's submissions to COLIEE 2021 presented in the paper [To Tune or Not To Tune? Zero-shot Models for Legal Case Entailment](https://dl.acm.org/doi/10.1145/3462757.3466103). There has been mounting evidence that pretrained language models fine-tuned on large and diverse supervised datasets can transfer well to a variety of out-of-domain tasks. In this work, we investigate this transfer ability to the legal domain. For that, we participated in the legal case entailment task of COLIEE 2021, in which we use such models with no adaptations to the target domain. Our submissions achieved the highest scores, surpassing the second-best submission by more than six percentage points. Our experiments confirm a counter-intuitive result in the new paradigm of pretrained language models: that given limited labeled data, models with little or no adaption to the target task can be more robust to changes in the data distribution and perform better on held-out datasets than models fine-tuned on it.


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


## How do I get the dataset?

Those who wish to use previous COLIEE data for a trial, please contact rabelo(at)ualberta.ca.


## How do I evaluate?

As our best model is a zero-shot one, we provide only the evaluation script.
- [Evaluation notebook](https://colab.research.google.com/drive/1bB4YiJm7_de0bsPJOWv9V0HYGj8_IzwU?usp=sharing) (Test set 2021)


## References

[1] [Document Ranking with a Pretrained Sequence-to-Sequence Model](https://arxiv.org/abs/2003.06713)

[2] [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654)

[3] [ICAIL '21: Proceedings of the Eighteenth International Conference on Artificial Intelligence and Law](https://dl.acm.org/doi/10.1145/3462757.3466103)

[4] [Proceedings of the Eigth International Competition on Legal Information Extraction/Entailment](https://sites.ualberta.ca/~rabelo/COLIEE2021/COLIEE2021proceedings.pdf)


## How do I cite this work?

~~~ {.xml
 @article{to_tune,
    title={To Tune or Not To Tune? Zero-shot Models for Legal Case Entailment},
    author={Moraes, Guilherme and Rodrigues, Ruan and Lotufo, Roberto and Nogueira, Rodrigo},
    journal={ICAIL '21: Proceedings of the Eighteenth International Conference on Artificial Intelligence and Law June 2021 Pages 295–300},
    url={https://dl.acm.org/doi/10.1145/3462757.3466103},
    year={2021}
}
~~~
