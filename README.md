## Introduction
Please read the original [pytorch bert documentation](https://github.com/huggingface/pytorch-pretrained-BERT) first. Specifically, install all the required packages (e.g. `pytorch_pretrained_bert`).

## Get Started
- The `run_natural_qa.py` is modified from `run.squad.py` in the `examples` folder. To run on natural questions (NQ) dataset, follow the [sample script](https://github.com/huggingface/pytorch-pretrained-BERT) that runs on SQUAD, change the python file name and data path. You may need to change other training configs to fit your machine (e.g. batch size)
- Now the `run_natural_qa.py` can barely run through a toy training process on a subset (now only 1 training file) of Google NQ dataset.

## Todos
- Load the entire training file (now only the first few entries)
- Load all training files (now only the first file)
- Change the last layer to output short answers, then use this output to get long answers.
- Write (modify) code to output evaluation result, use threshold to output no answer (if no answer should be found). Refer to the [official evaluation code](https://github.com/google-research-datasets/natural-questions/blob/master/nq_eval.py)
- Implement other techniques in the baseline.
- Get our own baseline results.
- Our novelties (Yes/No answers; multiple short answers; heuristics in evaluation; usage of long/short answer types, etc.).

## Memos
- The program may hang/crash when handling large inputs (each decompressed training file is around 5 GB)
- The traning files are currently arranged as `pytorch-pretrained-BERT/data/natural_questions/v1.0/train/nq-train-00.jsonl`. The dataset (compressed) is available online (40 - 50 GB). Remember to decompress before use.
- The output files are currently at `pytorch-pretrained-BERT/output/nq_1`.

## Debug
- To Follow the baseline implementation, problem. We need tomodify the source code of `pytorch_pretrained_bert`. 
  - Open `/home/peisheng/.local/lib/python3.6/site-packages/pytorch_pretrained_bert/tokenization.py` (change to your path)
  - Change line 78 from
```
never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
```
to
```
never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]", "[html_token]")):
```
  - Change line 102 from
```
ids.append(self.vocab[token])
``` 
  to 
```
if token in self.vocab.keys():
    ids.append(self.vocab[token])
else:
    ids.append(self.vocab['[UNK]'])
```

  - Change line 166 from
```
never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
```
to
```
never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]", "[html_token]")):
```

  and save it.
  - Replace the vocabulary with the augmented vocabulary. The augmented vocabulary is in `augmented_vocab`, the default one is at `/home/peisheng/.pytorch_pretrained_bert/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084` (change to your path). The detail of the vocab needs further discussion.

  
## Useful links
- [Official preprocessing code](https://github.com/google-research/language/tree/master/language/question_answering) in Tensorflow
