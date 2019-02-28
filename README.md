## Introduction
Please read the original [pytorch bert documentation](https://github.com/huggingface/pytorch-pretrained-BERT) first. Specifically, install all the required packages (e.g. `pytorch_pretrained_bert`).

## Get Started
- The `run_natural_qa.py` is modified from `run.squad.py` in the `examples` folder. To run on natural questions (NQ) dataset, follow the [sample script](https://github.com/huggingface/pytorch-pretrained-BERT) that runs on SQUAD, change the python file name and data path. You may need to change other training configs to fit your machine (e.g. batch size)
- Now the `run_natural_qa.py` can barely run through a toy training process on Google NQ dataset.

## Todos
- Load the entire training file (now only the first 5 entries)
- Load all training files (now only the first file)
- Load short answers (now only long answers)
- Change the last layer to output short answers, then use this output to get long answers.
- Write (modify) code to output evaluation result, use threshold to output no answer (if no answer should be found). Refer to the [official evaluation code](https://github.com/google-research-datasets/natural-questions/blob/master/nq_eval.py)
- Incorporate `index of paragraphs/tables/etc` into inputs, as stated in the [baseline paper](https://arxiv.org/abs/1901.08634) 
- Implement other techniques in the baseline.
- Get our own baseline results.
- Our novelties (Yes/No answers; multiple short answers; heuristics in evaluation; usage of long/short answer types, etc.).

## Memos
- The program may hang/crash when handling large inputs (each decompressed training file is around 5 GB)
- The traning files are currently arranged as `pytorch-pretrained-BERT/data/natural_questions/v1.0/train/nq-train-00.jsonl`. The dataset (compressed) is available online (40 - 50 GB). Remember to decompress before use.
- The output files are currently at `pytorch-pretrained-BERT/output/nq_1`.

## Debug
- The current `run_natural_qa.py` can run into *token outside vocabulary* problem. A temporal solution is to modify the source code of `pytorch_pretrained_bert`. 
  - Open `/home/peisheng/.local/lib/python3.6/site-packages/pytorch_pretrained_bert/tokenization.py` (your path)
  - Change line 102 from
                ```
                ids.append(self.vocab[token])
                ``` 
  to 
                ```
                if token in self.vocab.keys():\
                    ids.append(self.vocab[token])\
                else:\
                    ids.append(self.vocab['[UNK]'])
                ```
  and save it.

  - Please help debug this, prefrably by editing `run_natural_qa.py`, because `run_squad.py` works fine.
  
## Useful links
- [Official preprocessing code](https://github.com/google-research/language/tree/master/language/question_answering) in Tensorflow
