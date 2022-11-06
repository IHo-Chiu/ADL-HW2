
python run_glue.py \
  --train_file ./intent-classification-ntu-adl-hw1-fall-2022/train.json \
  --validation_file ./intent-classification-ntu-adl-hw1-fall-2022/eval.json \
  --test_file ${1} \
  --model_name_or_path ./intent \
  --max_seq_length 32 \
  --pad_to_max_length \
  --output_dir ./intent \
  --do_predict \
  --per_device_eval_batch_size 1 \
  --disable_tqdm False \
  --output_file ${2}

