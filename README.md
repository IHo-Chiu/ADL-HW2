# ADL-HW2 (r11922189)

## Environment
``` shell
# If you have conda, we recommend you to build a conda environment called "adl-hw2"
make
conda activate adl-hw2
pip install -r requirements.txt
# Otherwise
pip install -r requirements.in
```

## Download and run main model

``` shell
bash ./download.sh
bash ./run.sh ${1} ${2} ${3}
# bash ./run.sh ntu-adl-hw2-fall-2022/context.json ntu-adl-hw2-fall-2022/test.json results.csv
```
${1}: context file path
${2}: test file path
${3}: predict result path

## Reproduce other models results

### Question Answering Problem
``` shell
bash ./download_bert.sh
bash ./download_bert_no_pretrained.sh
bash ./run.sh ${1} ${2} ${3} ${4}
# bash ./run.sh ntu-adl-hw2-fall-2022/context.json ntu-adl-hw2-fall-2022/test.json results.csv bert
```

${1}: context file path
${2}: test file path
${3}: predict result path
${4}: option model: (bert, bert_no_pretrained, roberta)

### Intent Classification
``` shell
bash ./download_intent.sh
bash ./run_intent.sh ${1} ${2}
# bash ./run_intent.sh intent-classification-ntu-adl-hw1-fall-2022/test.json results.csv
```

${1}: test file path
${2}: predict result path

### Slot Tagging

need to install seqeval

``` shell
bash ./download_slot.sh
bash ./run_slot.sh ${1} ${2}
# bash ./run_slot.sh slot-tagging-ntu-adl-hw1-fall-2022/test.json results.csv
```

${1}: test file path
${2}: predict result path

## Train models

### Datasets
https://www.kaggle.com/competitions/ntu-adl-hw2-fall-2022

### Pretrained Models
https://huggingface.co/models?search=chinese

### Training Arguments Document
https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/trainer#transformers.TrainingArguments

### Content Selection

#### Train
``` shell
python run_swag.py \
  --train_file ./ntu-adl-hw2-fall-2022/train.json \
  --validation_file ./ntu-adl-hw2-fall-2022/valid.json \
  --context_file ./ntu-adl-hw2-fall-2022/context.json \
  --model_name_or_path hfl/chinese-roberta-wwm-ext \
  --cache_dir ./cache \
  --max_seq_length 512 \
  --pad_to_max_length \
  --output_dir ./roberta2/content \
  --do_train \
  --do_eval \
  --evaluation_strategy steps \
  --eval_steps 1000 \
  --save_steps 1000 \
  --logging_steps 1000 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --eval_accumulation_steps 2 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --warmup_ratio 0.1 \
  --disable_tqdm False \
  --metric_for_best_model accuracy \
  --load_best_model_at_end True
```

#### Evaluate
``` shell
python run_swag.py \
  --validation_file ./ntu-adl-hw2-fall-2022/valid.json \
  --context_file ./ntu-adl-hw2-fall-2022/context.json \
  --model_name_or_path ./roberta/content \
  --max_seq_length 512 \
  --pad_to_max_length \
  --output_dir ./roberta/content \
  --do_eval \
  --per_device_eval_batch_size 1 \
  --disable_tqdm False
```

#### Predict
``` shell
python run_swag.py \
  --test_file ./ntu-adl-hw2-fall-2022/test.json \
  --context_file ./ntu-adl-hw2-fall-2022/context.json \
  --model_name_or_path ./roberta/content \
  --max_seq_length 512 \
  --pad_to_max_length \
  --output_dir ./roberta/content \
  --do_predict \
  --per_device_eval_batch_size 1 \
  --disable_tqdm False \
  --output_file content_predict_roberta.json
```


### Question Answering

#### Train
``` shell
python run_qa.py \
  --train_file ./ntu-adl-hw2-fall-2022/train.json \
  --validation_file ./ntu-adl-hw2-fall-2022/valid.json \
  --context_file ./ntu-adl-hw2-fall-2022/context.json \
  --model_name_or_path hfl/chinese-roberta-wwm-ext-large \
  --cache_dir ./cache \
  --max_seq_length 512 \
  --pad_to_max_length \
  --output_dir ./roberta/qa \
  --do_train \
  --do_eval \
  --evaluation_strategy steps \
  --eval_steps 50 \
  --save_steps 50 \
  --logging_steps 50 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 8 \
  --eval_accumulation_steps 8 \
  --learning_rate 3e-5 \
  --num_train_epochs 10 \
  --warmup_ratio 0.1 \
  --disable_tqdm False \
  --metric_for_best_model accuracy \
  --load_best_model_at_end True
```

#### Evaluate
``` shell
python run_qa.py \
  --validation_file ./ntu-adl-hw2-fall-2022/valid.json \
  --context_file ./ntu-adl-hw2-fall-2022/context.json \
  --model_name_or_path ./roberta/qa \
  --max_seq_length 512 \
  --pad_to_max_length \
  --output_dir ./roberta/qa \
  --do_eval \
  --per_device_eval_batch_size 1 \
  --disable_tqdm False
```

#### Predict
``` shell
python run_qa.py \
  --test_file content_predict_roberta.json \
  --context_file ./ntu-adl-hw2-fall-2022/context.json \
  --model_name_or_path ./roberta/qa \
  --max_seq_length 512 \
  --pad_to_max_length \
  --output_dir ./roberta/qa \
  --do_predict \
  --per_device_eval_batch_size 1 \
  --disable_tqdm False \
  --output_file ./results_roberta.csv
```

### Intent Classification

#### Train
``` shell
python run_glue.py \
  --train_file ./intent-classification-ntu-adl-hw1-fall-2022/train.json \
  --validation_file ./intent-classification-ntu-adl-hw1-fall-2022/eval.json \
  --model_name_or_path bert-base-uncased \
  --cache_dir ./cache \
  --max_seq_length 32 \
  --pad_to_max_length \
  --output_dir ./intent \
  --do_train \
  --do_eval \
  --evaluation_strategy steps \
  --eval_steps 50 \
  --save_steps 50 \
  --logging_steps 50 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 8 \
  --eval_accumulation_steps 8 \
  --learning_rate 3e-5 \
  --num_train_epochs 5 \
  --warmup_ratio 0.1 \
  --disable_tqdm False \
  --metric_for_best_model accuracy \
  --load_best_model_at_end True
```

#### Predict
``` shell
python run_glue.py \
  --train_file ./intent-classification-ntu-adl-hw1-fall-2022/train.json \
  --validation_file ./intent-classification-ntu-adl-hw1-fall-2022/eval.json \
  --test_file ./intent-classification-ntu-adl-hw1-fall-2022/test.json \
  --model_name_or_path ./intent \
  --max_seq_length 32 \
  --pad_to_max_length \
  --output_dir ./intent \
  --do_predict \
  --per_device_eval_batch_size 1 \
  --disable_tqdm False \
  --output_file ./results_intent.csv
```

### Slot Tagging

#### Train
``` shell
python run_ner.py \
  --train_file ./slot-tagging-ntu-adl-hw1-fall-2022/train.json \
  --validation_file ./slot-tagging-ntu-adl-hw1-fall-2022/eval.json \
  --model_name_or_path bert-base-uncased \
  --cache_dir ./cache \
  --max_seq_length 35 \
  --pad_to_max_length \
  --output_dir ./slot \
  --do_train \
  --do_eval \
  --evaluation_strategy steps \
  --eval_steps 50 \
  --save_steps 50 \
  --logging_steps 50 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 8 \
  --eval_accumulation_steps 8 \
  --learning_rate 3e-5 \
  --num_train_epochs 5 \
  --warmup_ratio 0.1 \
  --disable_tqdm False \
  --metric_for_best_model accuracy \
  --load_best_model_at_end True
```

#### Predict
``` shell
python run_ner.py \
  --train_file ./slot-tagging-ntu-adl-hw1-fall-2022/train.json \
  --validation_file ./slot-tagging-ntu-adl-hw1-fall-2022/eval.json \
  --test_file ./slot-tagging-ntu-adl-hw1-fall-2022/test.json \
  --model_name_or_path ./slot \
  --max_seq_length 35 \
  --pad_to_max_length \
  --output_dir ./slot \
  --do_predict \
  --per_device_eval_batch_size 1 \
  --disable_tqdm False \
  --output_file ./results_slot.csv
```