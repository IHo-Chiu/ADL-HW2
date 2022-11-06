
if [ -z "$4" ]
  then
    python run_swag.py \
      --test_file ${2} \
      --context_file ${1} \
      --model_name_or_path ./roberta/content \
      --max_seq_length 512 \
      --pad_to_max_length \
      --output_dir ./roberta/content \
      --do_predict \
      --per_device_eval_batch_size 1 \
      --disable_tqdm False \
      --output_file temp.json


    python run_qa.py \
      --test_file temp.json \
      --context_file ${1} \
      --model_name_or_path ./roberta/qa \
      --max_seq_length 512 \
      --pad_to_max_length \
      --output_dir ./roberta/qa \
      --do_predict \
      --per_device_eval_batch_size 1 \
      --disable_tqdm False \
      --output_file ${3}
      
  else
    python run_swag.py \
      --test_file ${2} \
      --context_file ${1} \
      --model_name_or_path ./${4}/content \
      --max_seq_length 512 \
      --pad_to_max_length \
      --output_dir ./${4}/content \
      --do_predict \
      --per_device_eval_batch_size 1 \
      --disable_tqdm False \
      --output_file temp.json


    python run_qa.py \
      --test_file temp.json \
      --context_file ${1} \
      --model_name_or_path ./${4}/qa \
      --max_seq_length 512 \
      --pad_to_max_length \
      --output_dir ./${4}/qa \
      --do_predict \
      --per_device_eval_batch_size 1 \
      --disable_tqdm False \
      --output_file ${3}
fi
  
rm temp.json

